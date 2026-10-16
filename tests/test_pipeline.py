"""Offline fixtures only: no video decode, network, or build entrypoint invocation."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from streamdrivebench import protocol as W
from streamdrivebench.pipeline import Runner,export
from streamdrivebench.common import validate_schema,load_json


def question(family):
    return dict(question='Which observable state?',choices=['a','b']+(['SILENT'] if family=='ETP' else []),
        target_description='Visible car',predicate='begins a turn',answer_slot='direction',rationale_from_prefix='visible in prefix')


def candidate(family,t):
    return dict(family=family,query_second=t,horizon_seconds=3 if family=='AD' else 0,question=question(family))


class Data:
    def valid(self,eid,frames):return all(0<=f<60 for f in frames)
    def image_path(self,eid,f):return Path(str(f)+'.jpg')


class Client:
    def __init__(self):self.calls=[]
    def call(self,prompt,payload,images,schema):
        self.calls.append(deepcopy(payload));stage=payload['stage']
        if stage=='event_discovery':
            result=dict(events=[] if payload['ownership_start'] else [dict(actor='car',behavior='turn',
                before_state='straight',after_state='turning',approximate_transition=10,evidence_frames=[9,10])],coverage_notes='fixture')
        elif stage=='qa_author':
            result=dict(question_sets=[dict(event_id=e['event_id'],questions=[candidate('HGI',12),candidate('AD',8),candidate('ETP',5)],omissions=[]) for e in payload['confirmed_events']])
        elif stage=='window_boundary':
            result=dict(valid=True,actor='car',behavior='turn',before_state='straight',after_state='turning',
                first_observable=10,last_absent=9,evidence_frames=[9,10],visibility_clear=True,difficulty_factors=[],reason='fixture')
        elif stage in ('hgi_current','hgi_full','ad_future'):
            result=dict(valid=stage!='hgi_current',answer='' if stage=='hgi_current' else 'a',
                evidence_frames=[12] if stage=='hgi_current' else ([9,12] if stage=='hgi_full' else [11]),
                target_identifiable=True,unambiguous=stage!='hgi_current',future_presupposition=False,
                history_necessary=stage=='hgi_full',reason='fixture')
        elif stage=='ad_prefix':result=dict(valid=True,target_identifiable=True,unambiguous=True,future_presupposition=False,reason='fixture')
        elif stage=='injection':result=dict(valid=True,target_identifiable=True,driving_state_change=True,
            single_atomic_condition=True,condition_already_true=False,future_presupposition=False,reason='fixture')
        elif stage=='monitor':result=dict(valid=True,target_consistent=True,visibility_sufficient=True,
            first_observable=10,last_absent=9,answer='a',evidence_frames=[5,9,10],event_absent_throughout=False,reason='fixture')
        elif stage=='association':result=dict(answer_supported=True,options_cover_observation=True,same_actor=True,
            same_behavior_episode=True,same_atomic_transition=True,semantic_duplicate_indices=[],relation='same',reason='fixture')
        elif stage=='etp_commitment':
            now=payload['query_second'];result=dict(target_identifiable=True,evidence_decidable=True,
                may_answer_now=now==10,answer='a' if now==10 else 'SILENT',evidence_frames=[now],reason='fixture')
        else:raise AssertionError(stage)
        validate_schema(result,schema)
        return result


def test_discovery_windows_partition_without_context_overlap():
    assert list(W.windows(65))==[(0,30),(30,60),(60,65)]
    assert [f for s,e in W.windows(65) for f in range(s,e)]==list(range(65))


def test_etp_crosses_discovery_boundary_but_requires_full_monitor():
    assert not W.preflight(candidate('ETP',29),0,30,60)
    assert 'incomplete_monitor' in W.preflight(candidate('ETP',30),30,60,60)
    assert 'forecast_outside_window' in W.preflight(candidate('AD',28),0,30,60)
    assert min(W.prefix(30,40))==30


def test_three_family_export_and_zero_call_window_resume(tmp_path):
    ep=dict(episode_id='fixture',num_frames=60,source_sequence='set01')
    client=Client();runner=Runner(tmp_path,ep,Data(),client)
    assert runner.max_events==1
    windows=runner.run();report=export(tmp_path,ep,windows,list(runner.trace.values()))
    assert report['accepted_by_family']=={'HGI':1,'AD':1,'ETP':1}
    assert report['causal_commitment_audit']['passed']==1
    proposals=[c for c in client.calls if c['stage']=='event_discovery']
    assert [p['image_frames'] for p in proposals]==[list(range(30)),list(range(30,60))]
    assert next(c for c in client.calls if c['stage']=='monitor')['image_frames']==list(range(5,36))
    for c in client.calls:
        if c['stage'] in ('hgi_current','hgi_full','ad_prefix','injection','etp_commitment'):
            assert max(c['image_frames'])<=c['query_second']
            assert not set(c).intersection({'event','label','answer','transition_second'})
            assert 'rationale_from_prefix' not in c['question']
    public=load_json(tmp_path/'model_inputs.json')
    etp=next(q for q in public if q['family']=='ETP')
    assert etp['input_policy']['online_monitor_end_second']==35
    assert all('evaluator_only' not in q for q in public)
    second=Client();resumed=Runner(tmp_path,ep,Data(),second)
    assert resumed.run()==windows and not second.calls


def test_hgi_current_sufficient_stops_history_call(tmp_path):
    client=Client();original=client.call
    def call(prompt,payload,images,schema):
        value=original(prompt,payload,images,schema)
        if payload['stage']=='hgi_current':value.update(valid=True,unambiguous=True,answer='a')
        return value
    client.call=call
    runner=Runner(tmp_path,dict(episode_id='fixture',num_frames=60),Data(),client)
    _,errors,_=runner.verify(candidate('HGI',12),0,30)
    assert errors==['current_frame_sufficient'] and len(client.calls)==1


def test_invalid_tail_etp_sends_no_api(tmp_path):
    client=Client();runner=Runner(tmp_path,dict(episode_id='fixture',num_frames=60),Data(),client)
    _,errors,_=runner.verify(candidate('ETP',40),30,60)
    assert errors==['incomplete_monitor'] and not client.calls


def test_causal_request_rejects_future_and_answer_metadata(tmp_path):
    client=Client();runner=Runner(tmp_path,dict(episode_id='fixture',num_frames=60),Data(),client)
    with pytest.raises(ValueError,match='leakage'):
        runner.call('',dict(stage='ad_prefix',image_frames=[10,11],query_second=10),{})
    with pytest.raises(ValueError,match='leakage'):
        runner.call('',dict(stage='injection',image_frames=[10],query_second=10,answer='a'),{})
    assert not client.calls


def test_all_boundaries_confirmed_before_shared_author(tmp_path):
    client=Client();original=client.call
    def call(prompt,payload,images,schema):
        result=original(prompt,payload,images,schema)
        if payload['stage']=='event_discovery':
            other=deepcopy(result['events'][0]);other['actor']='second car'
            result['events'].append(other)
        elif payload['stage']=='window_boundary' and payload['proposal']['actor']=='second car':
            result['actor']='second car';result['first_observable']=20
            result['last_absent']=19;result['evidence_frames']=[19,20]
        return result
    client.call=call
    runner=Runner(tmp_path,dict(episode_id='fixture',num_frames=60,source_sequence='set01'),Data(),client,max_events=2)
    result=runner.window(0,30)
    stages=[c['stage'] for c in client.calls]
    assert stages[:4]==['event_discovery','window_boundary','window_boundary','qa_author']
    assert stages.count('qa_author')==1 and len(result['events'])==2
    author=client.calls[3]
    assert len(author['confirmed_events'])==2
    assert all(e['boundary_check']['valid'] for e in author['confirmed_events'])
    assert all('questions' not in e for e in author['confirmed_events'])
    assert client.calls[0]['max_events']==2
    assert all('duplicate_model_input' in q['errors'] for q in result['candidates'][3:])


def test_rejected_event_has_no_author_call(tmp_path):
    client=Client();original=client.call
    def call(prompt,payload,images,schema):
        result=original(prompt,payload,images,schema)
        if payload['stage']=='window_boundary':result['valid']=False
        return result
    client.call=call
    result=Runner(tmp_path,dict(episode_id='fixture',num_frames=60),Data(),client).window(0,30)
    assert not result['events'] and not result['candidates']
    assert [c['stage'] for c in client.calls]==['event_discovery','window_boundary']


@pytest.mark.parametrize('fault',['unknown_event','missing_family','blank_omission','duplicate_family'])
def test_author_cannot_invent_events_or_silently_omit_families(tmp_path,fault):
    client=Client();original=client.call
    def call(prompt,payload,images,schema):
        result=original(prompt,payload,images,schema)
        if payload['stage']=='qa_author':
            row=result['question_sets'][0]
            if fault=='unknown_event':row['event_id']='not-confirmed'
            if fault=='missing_family':row['questions'].pop()
            if fault=='blank_omission':
                row['questions'].pop();row['omissions']=[dict(family='ETP',reason=' ')]
            if fault=='duplicate_family':row['questions'].append(deepcopy(row['questions'][0]))
        return result
    client.call=call
    runner=Runner(tmp_path,dict(episode_id='fixture',num_frames=60,source_sequence='set01'),Data(),client)
    with pytest.raises(ValueError):runner.window(0,30)
    assert not any(c['stage']=='hgi_current' for c in client.calls)


def test_justified_omission_preserved_in_report(tmp_path):
    client=Client();original=client.call
    def call(prompt,payload,images,schema):
        result=original(prompt,payload,images,schema)
        if payload['stage']=='qa_author':
            row=result['question_sets'][0];row['questions'].pop()
            row['omissions']=[dict(family='ETP',reason='No identifiable target before the transition.')]
        return result
    client.call=call
    ep=dict(episode_id='fixture',num_frames=60,source_sequence='set01')
    runner=Runner(tmp_path,ep,Data(),client);windows=runner.run()
    report=export(tmp_path,ep,windows,list(runner.trace.values()))
    assert report['author_omissions'][0]['family']=='ETP'
    assert report['accepted_by_family']=={'HGI':1,'AD':1}
