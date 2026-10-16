from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import pytest
from streamdrivebench import specific
from streamdrivebench.budget import Budget, BudgetExhausted
from streamdrivebench.common import dump_json, load_json
from streamdrivebench import protocol as W
from test_pipeline import Client as FixtureClient, candidate, Data
from streamdrivebench.pipeline import Runner, export
from test_portable import video


def entries(n=4,seconds=90):
    return [dict(episode_id=str(i),clip=dict(duration_seconds=seconds,file=f'{i}.mp4')) for i in range(n)]


def test_inventory_round_robin_reproducibility_and_short_tail():
    es=entries();order=specific.window_order(es,7)
    assert order==specific.window_order(es,7)
    assert order!=specific.window_order(es,8)
    assert len(order)==len({specific.key(t) for t in order})==12
    for i in range(0,12,4):assert len({t['episode_id'] for t in order[i:i+4]})==4
    assert not specific.window_order(entries(1,1),7)
    assert specific.eligible(dict(start=30,end=40),40)==['HGI','AD']
    assert specific.eligible(dict(start=0,end=30),40)==['HGI','AD','ETP']


def test_reservations_do_not_exceed_targets_and_resume_pending():
    order=specific.window_order(entries(10),2);assignments=[];counts=Counter()
    targets=dict(HGI=2,AD=1,ETP=3);lengths={str(i):90 for i in range(10)}
    batch=specific.allocate(order,assignments,{},targets,counts,24,lengths,100)
    reserved=Counter(f for t in batch for f in t['families'])
    assert reserved==targets
    assert len({t['episode_id'] for t in batch})==len(batch)
    assert specific.allocate(order,assignments,{},targets,counts,24,lengths,100)==batch
    completed={specific.key(t):{} for t in batch}
    assert not specific.allocate(order,assignments,completed,targets,Counter(targets),24,lengths,100)
    # All failed quality checks: reservations released, different windows fill the deficits.
    again=specific.allocate(order,assignments,completed,targets,Counter(),24,lengths,100)
    assert Counter(f for t in again for f in t['families'])==targets
    assert not {specific.key(t) for t in batch}&{specific.key(t) for t in again}


def test_budget_persists_hard_attempt_limit_across_threads_and_restarts(tmp_path):
    path=tmp_path/'budget.json';budget=Budget(path,3)
    def reserve(_):
        try:budget.reserve();budget.finish({'total_tokens':7});return True
        except BudgetExhausted:return False
    with ThreadPoolExecutor(8) as pool:assert sum(pool.map(reserve,range(16)))==3
    assert load_json(path)==dict(attempts_reserved=3,reported_tokens=21,unsettled_attempts=0)
    with pytest.raises(BudgetExhausted):Budget(path,3).reserve()
    more=Budget(path,4);more.reserve();more.finish()
    with pytest.raises(BudgetExhausted):Budget(path,100,21).reserve()


def install_client(monkeypatch):
    calls=[]
    class Client(FixtureClient):
        def __init__(self,*a,**kw):super().__init__();self.budget=kw['budget']
        def call(self,prompt,payload,images,schema):
            self.budget.reserve()
            result=super().call(prompt,payload,images,schema)
            calls.append(deepcopy(payload));self.budget.finish({'total_tokens':10})
            if payload['stage']=='qa_author':
                for item in result['question_sets']:
                    item['questions']=[q for q in item['questions'] if q['family'] in payload['requested_families']]
                    for q in item['questions']:
                        q['history_start_second']={'HGI':9,'AD':7,'ETP':4}[q['family']]
            return result
    monkeypatch.setattr(specific,'Client',Client)
    monkeypatch.setenv('OPENAI_API_KEY','offline-only')
    return calls


def test_sparse_end_to_end_exact_quotas_short_history_resume(tmp_path,monkeypatch):
    calls=install_client(monkeypatch)
    for i in range(4):video(tmp_path/f'in/{i}.mp4',60,1)
    args=['--dataset','bdd100k','--video-dir',str(tmp_path/'in'),'--output',str(tmp_path/'out'),
          '--model','fixture','--targets','2','1','1','--workers','4']
    specific.main(args)
    summary=load_json(tmp_path/'out/summary.json')
    assert summary['by_family']=={'HGI':2,'AD':1,'ETP':1}
    assert summary['build_state']=='targets_met'
    assert summary['completed_windows']<8
    r=tmp_path/'out/release';rows=load_json(r/'queries_all.json')
    for row in rows:
        assert row['input_policy']['history_start_second']=={'HGI':9,'AD':7,'ETP':4}[row['family']]
        if row['family']=='ETP':
            assert row['input_policy']['online_monitor_end_second']==35
            assert list(row['model_input']['choices'].values())[-1]=='SILENT'
    assert all(min(p['image_frames'])>=9 for p in calls if p['stage']=='hgi_full')
    assert all(min(p['image_frames'])>=7 for p in calls if p['stage']=='ad_prefix')
    assert all(min(p['image_frames'])>=4 for p in calls if p['stage']=='injection')
    assert all(len(p['confirmed_events'])==1 for p in calls if p['stage']=='qa_author')
    before=len(calls);specific.main(args);assert len(calls)==before
    assert load_json(tmp_path/'out/summary.json')['by_family']==summary['by_family']


def test_budget_stop_publishes_explicit_partial_and_resume(tmp_path,monkeypatch):
    install_client(monkeypatch);video(tmp_path/'in/v.mp4',60,1)
    args=['--dataset','bdd100k','--video-dir',str(tmp_path/'in'),'--output',str(tmp_path/'out'),
          '--targets','1','1','1','--max-api-attempts','1']
    with pytest.raises(SystemExit) as exc:specific.main(args)
    assert exc.value.code==2
    s=load_json(tmp_path/'out/summary.json')
    assert s['build_state']=='budget_exhausted' and s['queries']==0
    assert s['remaining']=={'HGI':1,'AD':1,'ETP':1}
    before=load_json(tmp_path/'out/api_budget.json')['attempts_reserved']
    with pytest.raises(SystemExit):specific.main(args)
    assert load_json(tmp_path/'out/api_budget.json')['attempts_reserved']==before
    args[-1]='100';specific.main(args)
    assert load_json(tmp_path/'out/summary.json')['build_state']=='targets_met'


def test_exhaustion_does_not_relax_quality_or_require_full_history(tmp_path,monkeypatch):
    calls=install_client(monkeypatch);video(tmp_path/'in/v.mp4',1,10)
    args=['--dataset','bdd100k','--video-dir',str(tmp_path/'in'),'--output',str(tmp_path/'out')]
    with pytest.raises(SystemExit):specific.main(args)
    assert not calls
    assert load_json(tmp_path/'out/summary.json')['build_state']=='candidate_windows_exhausted'
    q=candidate('HGI',12);q['history_start_second']=9
    assert not W.preflight(q,0,30,60)
    q['history_start_second']=12
    assert 'no_history' in W.preflight(q,0,30,60)
    q['history_start_second']=-1
    assert 'invalid_history_window' in W.preflight(q,0,30,60)
    q=candidate('AD',8);q['history_start_second']=8
    assert not W.preflight(q,0,30,60)


def test_presets_match_checked_in_plan():
    from pathlib import Path
    plan=load_json(Path(__file__).parents[1]/'configs/specific_targets.json')
    for name,p in specific.PRESETS.items():
        for k,v in p.items():assert plan['datasets'][name][k]==v
    all_targets=[p['targets'] for p in plan['datasets'].values()]+[p['targets'] for p in plan['already_built_sample_targets'].values()]
    assert {f:sum(t[f] for t in all_targets) for f in specific.FAMILIES}==dict(HGI=2318,AD=2122,ETP=1560)


def test_dry_run_no_key_no_output_and_plan_freeze(tmp_path,monkeypatch):
    video(tmp_path/'in/v.mp4',24,10);monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    args=['--dataset','bdd100k','--video-dir',str(tmp_path/'in'),'--output',str(tmp_path/'out')]
    specific.main(args+['--dry-run']);assert not (tmp_path/'out').exists()
    install_client(monkeypatch)
    with pytest.raises(SystemExit):specific.main(args+['--targets','1','0','0'])
    with pytest.raises(ValueError,match='Frozen'):specific.main(args+['--targets','2','0','0'])
