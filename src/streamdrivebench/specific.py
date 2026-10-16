"""Quota-driven sparse generation, with durable window assignments and reservations."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import math
import os
from pathlib import Path
import random
import threading
from urllib.parse import urlsplit
from .budget import Budget, BudgetExhausted
from .common import dump_json, load_json, freeze, file_hash
from .media import discover_clips, prepare_episode, Dataset
from .client import Client
from .pipeline import Runner, export
from .release import publish
from .cli import DEFAULT_BASE_URL, usage

FAMILIES=('HGI','AD','ETP')
PRESETS={
    'bdd100k':dict(videos=50,targets=dict(HGI=19,AD=18,ETP=13),max_api_attempts=1000),
    'opendv-youtube':dict(videos=352,targets=dict(HGI=1545,AD=1415,ETP=1040),max_api_attempts=60000),
    'boreas':dict(videos=125,targets=dict(HGI=290,AD=265,ETP=195),max_api_attempts=12000),
}


def window_order(entries, seed):
    """Equal video turns, uniform shuffled windows within each video; no API."""
    rng=random.Random(seed);videos=sorted(entries,key=lambda e:e['episode_id']);rng.shuffle(videos)
    queues={}
    for e in videos:
        duration=e['clip']['duration_seconds']
        if duration is None or not math.isfinite(duration) or duration<=0:
            raise ValueError('Missing/invalid duration: prepare or remux the video before planning: '+e['clip']['file'])
        n=math.ceil(duration);starts=list(range(0,n,30))
        random.Random(f"{seed}:{e['episode_id']}").shuffle(starts)
        queues[e['episode_id']]=[dict(episode_id=e['episode_id'],start=s,end=min(s+30,n)) for s in starts if min(s+30,n)-s>=2]
    result=[]
    while any(queues.values()):
        for e in videos:
            q=queues[e['episode_id']]
            if q:result.append(q.pop())
    return result


def key(task):return f"{task['episode_id']}:{task['start']}"


def eligible(task, n):
    # AD needs a transition and a query with a 3-second future inside discovery.
    # ETP always keeps the full future monitor, even when history is short.
    return [f for f in FAMILIES if (f!='AD' or task['end']-task['start']>=4)
            and (f!='ETP' or task['start']+30<n)]


def allocate(order, assignments, completed, targets, counts, batch_size, lengths, max_windows):
    """Reserve one slot per requested family. At most one window per video per batch."""
    reserved=Counter();chosen=[];active_videos=set();known={key(t) for t in assignments}
    for task in assignments:
        if key(task) not in completed:
            reserved.update(task['families'])
    # Unfinished assignments retain their family reservation across crashes.
    for task in assignments:
        if key(task) not in completed and task['episode_id'] not in active_videos and len(chosen)<batch_size:
            chosen.append(task);active_videos.add(task['episode_id'])
    for task in order:
        if len(chosen)>=batch_size or len(assignments)>=max_windows:break
        if key(task) in known or task['episode_id'] in active_videos:continue
        families=[f for f in eligible(task,lengths[task['episode_id']]) if counts[f]+reserved[f]<targets[f]]
        if not families:continue
        new=dict(task,families=families);assignments.append(new);known.add(key(new));chosen.append(new)
        active_videos.add(task['episode_id']);reserved.update(families)
    return chosen


def validate_result(task, result):
    if result['start']!=task['start'] or result['end_exclusive']!=task['end']:
        raise ValueError('Window checkpoint mismatch')
    counts=Counter(q['family'] for q in result['accepted'])
    if set(counts)-set(task['families']) or any(v>1 for v in counts.values()):
        raise ValueError('Window exceeded reserved QA quota')
    if any(q['errors'] for q in result['accepted']):raise ValueError('Invalid QA accepted')


def checkpoint(root, task):
    return root/'videos'/task['episode_id']/'episodes'/task['episode_id']/'windows'/f"{task['start']:06d}.json"


def collect(root, assignments):
    completed={};counts=Counter({f:0 for f in FAMILIES})
    for task in assignments:
        path=checkpoint(root,task)
        if path.exists():
            value=load_json(path);validate_result(task,value);completed[key(task)]=value
            counts.update(q['family'] for q in value['accepted'])
    return completed,counts


def publish_partial(entries, root, assignments, completed, targets, counts, state):
    ready=[]
    for entry in entries:
        windows=[completed[key(t)] for t in assignments if t['episode_id']==entry['episode_id'] and key(t) in completed]
        if not windows:continue
        raw=Path(entry['raw']);ep=load_json(raw/'episodes_metadata.json')['episodes'][0]
        audit=raw/'episodes'/entry['episode_id']/'request_audit.json'
        trace=load_json(audit) if audit.exists() else []
        export(raw,ep,sorted(windows,key=lambda w:w['start']),trace,allow_sparse=True);ready.append(entry)
    build_metadata=dict(build_state=state,targets=targets,remaining={f:targets[f]-counts[f] for f in FAMILIES},
        discovery_policy='sparse_quota_reserved_windows')
    result=publish(ready,root/'release',build_metadata=build_metadata)
    attempted=Counter(f for t in assignments if key(t) in completed for f in t['families'])
    rates={f:counts[f]/attempted[f] if attempted[f] else None for f in FAMILIES}
    estimates={f:(math.ceil((targets[f]-counts[f])/rates[f]) if rates[f] else None)
               for f in FAMILIES if counts[f]<targets[f]}
    summary=dict(result,build_state=state,accepted_per_requested_window=rates,
        estimated_additional_windows_by_family=estimates,targets=targets,remaining={f:targets[f]-counts[f] for f in FAMILIES},
        inventoried_selected_videos=len(entries),scanned_videos=len(ready),
        contributing_videos=sum(v['queries']>0 for v in result['per_video']),
        assigned_windows=len(assignments),completed_windows=len(completed),
        quality_status='automated_checks_only_pending_human_review',
        note='Counts are ceilings; shortfalls are explicit. Video counts are candidates, not guaranteed QA coverage.')
    dump_json(summary,root/'summary.json');dump_json(summary,root/'status.json')
    progress={k:summary[k] for k in ('build_state','by_family','remaining','assigned_windows','completed_windows','contributing_videos')}
    print('QUOTA_PROGRESS '+json.dumps(progress,ensure_ascii=False),flush=True)
    return summary


def run(args, entries, order, root, source, targets, config):
    assignments_path=root/'window_assignments.json'
    assignments=load_json(assignments_path) if assignments_path.exists() else []
    completed,counts=collect(root,assignments)
    if any(counts[f]>targets[f] for f in FAMILIES):raise ValueError('Existing results exceed target')
    budget=Budget(root/'api_budget.json',args.max_api_attempts,args.max_reported_tokens)
    lookup={e['episode_id']:e for e in entries};lengths={e['episode_id']:math.ceil(e['clip']['duration_seconds']) for e in entries}
    runners={};runners_lock=threading.Lock()
    def work(task):
        eid=task['episode_id'];entry=lookup[eid];raw=Path(entry['raw'])
        with runners_lock:runner=runners.get(eid)
        if runner is None:
            ep=prepare_episode(dict(entry['clip'],sha256=entry['source_sha256']),source,raw)
            dump_json(dict(episodes=[ep]),raw/'episodes_metadata.json')
            data=Dataset(raw/'source_1fps',[ep])
            client=Client(root/'api_cache/calls'/eid,config,data,eid,threading.Semaphore(1),0,budget=budget)
            runner=Runner(raw,ep,data,client,max_events=1)
            with runners_lock:runners[eid]=runner
        # Incorrect container duration must not silently drop or invent windows.
        if runner.n!=lengths[eid]:
            raise ValueError('Decoded duration disagrees with inventory; remux source and use a new output: '+entry['clip']['file'])
        value=runner.window(task['start'],task['end'],task['families'])
        validate_result(task,value);dump_json(value,checkpoint(root,task));return value
    failure=None;state='running'
    try:
        while not all(counts[f]==targets[f] for f in FAMILIES):
            tasks=allocate(order,assignments,completed,targets,counts,args.batch_size,lengths,args.max_windows)
            if not tasks:
                state='window_limit_reached' if len(assignments)>=args.max_windows and args.max_windows<len(order) else 'candidate_windows_exhausted';break
            dump_json(assignments,assignments_path)  # Reserve durably BEFORE starting API work.
            batch_errors=[]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                # Only workers tasks are submitted; on failure no queued work starts.
                pending=iter(tasks);active={}
                def submit():
                    try:task=next(pending)
                    except StopIteration:return
                    active[pool.submit(work,task)]=task
                for _ in range(min(args.workers,len(tasks))):submit()
                while active:
                    future=next(as_completed(active));task=active.pop(future)
                    try:future.result()
                    except Exception as exc:batch_errors.append(exc)
                    if not batch_errors:submit()
            completed,counts=collect(root,assignments)
            publish_partial(entries,root,assignments,completed,targets,counts,'running')
            if batch_errors:
                failure=next((e for e in batch_errors if not isinstance(e,BudgetExhausted)),batch_errors[0])
                state='budget_exhausted' if isinstance(failure,BudgetExhausted) else 'failed';break
        else:state='targets_met'
    except BaseException:
        state='interrupted_or_failed'
        raise
    finally:
        completed,counts=collect(root,assignments)
        result=publish_partial(entries,root,assignments,completed,targets,counts,state)
        usage(root)
    if failure and not isinstance(failure,BudgetExhausted):raise failure
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=PRESETS)
    p.add_argument('--video-dir',required=True,type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--model',default=os.environ.get('OPENAI_MODEL','gpt-6.1-sol'))
    p.add_argument('--api-key',help='Prefer OPENAI_API_KEY')
    p.add_argument('--base-url',default=os.environ.get('OPENAI_BASE_URL',DEFAULT_BASE_URL))
    p.add_argument('--reasoning-effort',default='low',choices=['none','minimal','low','medium','high','xhigh'])
    p.add_argument('--targets',type=int,nargs=3,metavar=('HGI','AD','ETP'),help='Override preset ceilings')
    p.add_argument('--video-limit',type=int,help='Seeded subset; defaults to table count')
    p.add_argument('--seed',type=int,default=20261016)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=24)
    p.add_argument('--max-windows',type=int,help='Default min(available, 3 * largest family target); persistent scan ceiling')
    p.add_argument('--max-api-attempts',type=int,help='Persistent hard request-attempt ceiling, including retries')
    p.add_argument('--max-reported-tokens',type=int,default=0,help='Optional soft stop; in-flight calls and unreported usage can exceed it')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv);preset=PRESETS[a.dataset]
    targets=dict(zip(FAMILIES,a.targets)) if a.targets is not None else preset['targets'].copy()
    a.video_limit=preset['videos'] if a.video_limit is None else a.video_limit
    a.max_api_attempts=preset['max_api_attempts'] if a.max_api_attempts is None else a.max_api_attempts
    if min(targets.values())<0 or sum(targets.values())<1 or min(a.workers,a.batch_size,a.video_limit,a.max_api_attempts)<1 or a.max_reported_tokens<0:
        p.error('Limits must be positive; family targets/token stop may be zero')
    if a.max_windows is not None and a.max_windows<1:p.error('max-windows must be positive')
    parsed=urlsplit(a.base_url)
    if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        p.error('Base URL must be plain HTTP(S), without credentials/query/fragment')
    source=a.video_dir.resolve();root=(a.output or Path('outputs/specific')/a.dataset).resolve()
    if source==root or source in root.parents or root in source.parents:p.error('Video/output must be independent trees')
    clips=discover_clips(source);found=len(clips)
    random.Random(a.seed).shuffle(clips);clips=clips[:a.video_limit]
    for c in clips:c['episode_id']=a.dataset+'-'+c['episode_id']
    entries=[dict(clip=c,episode_id=c['episode_id'],raw=str(root/'videos'/c['episode_id'])) for c in clips]
    order=window_order(entries,a.seed)
    a.max_windows=a.max_windows if a.max_windows is not None else min(len(order),3*max(targets.values()))
    print(json.dumps(dict(dataset=a.dataset,found_videos=found,selected_videos=len(clips),available_windows=len(order),
        max_windows=a.max_windows,targets=targets,max_api_attempts=a.max_api_attempts,
        max_reported_tokens=a.max_reported_tokens,workers=a.workers,output=str(root)),ensure_ascii=False),flush=True)
    if a.dry_run:return
    if a.api_key:os.environ['OPENAI_API_KEY']=a.api_key
    if not os.environ.get('OPENAI_API_KEY'):p.error('Provide OPENAI_API_KEY; no API request sent')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for e in entries:e['source_sha256']=file_hash(source/e['clip']['file'])
        config=dict(model=a.model,effort=a.reasoning_effort,base_url=a.base_url,max_output_tokens=6500)
        freeze(root/'build_plan.json',dict(version='specific-v1',dataset=a.dataset,source=str(source),entries=entries,
            targets=targets,seed=a.seed,config=config,window_order=order,
            code_hashes={f.name:file_hash(f) for f in Path(__file__).parent.glob('*.py')}))
        dump_json(dict(workers=a.workers,batch_size=a.batch_size,max_windows=a.max_windows,
            max_api_attempts=a.max_api_attempts,max_reported_tokens=a.max_reported_tokens),root/'runtime_limits.json')
        result=run(a,entries,order,root,source,targets,config)
        print('FINAL_QA_COUNTS '+json.dumps(result['by_family'])+' state='+result['build_state'],flush=True)
        if result['build_state']!='targets_met':raise SystemExit(2)


if __name__=='__main__':main()
