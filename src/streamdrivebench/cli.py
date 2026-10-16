"""Portable, resumable multi-video orchestration for StreamDriveBench."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from .common import dump_json,load_json,freeze,file_hash
from .media import discover_clips
from .pipeline import build
from .release import publish

DEFAULT_BASE_URL='https://www.hongwangwang.qd.je/v1'


def usage(root):
    counts={};totals={}
    for f in (root/'api_cache').glob('calls/*/attempts/*.json'):
        item=load_json(f);state=item['status'];counts[state]=counts.get(state,0)+1
        for k,v in item.get('usage',{}).items():
            if type(v) in (int,float):totals[k]=totals.get(k,0)+v
    result=dict(attempt_status=counts,provider_reported_tokens=totals,
        note='Transport failures can incur unreported costs. Reasoning tokens are included in output_tokens, when reported.')
    dump_json(result,root/'token_usage.json');return result


def execute(entries,workers,work,update):
    pending=iter(entries);done=[];failures=[];active={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def submit():
            try:e=next(pending)
            except StopIteration:return False
            active[pool.submit(work,e)]=e;return True
        for _ in range(workers):
            if not submit():break
        update(done,list(active.values()),failures)
        while active:
            finished,_=wait(active,return_when=FIRST_COMPLETED)
            for f in finished:
                e=active.pop(f)
                try:f.result();done.append(e)
                except Exception as exc:failures.append(dict(video=e['clip']['file'],error_type=type(exc).__name__,message=str(exc)))
            if not failures:
                for _ in finished:
                    if not submit():break
            update(done,list(active.values()),failures)
    return done,failures


def main(argv=None):
    p=argparse.ArgumentParser(description='Discover/confirm driving events, generate and audit HGI/AD/ETP, then publish globally balanced QA.')
    p.add_argument('--video-dir',type=Path,required=True)
    p.add_argument('--model',default=os.environ.get('OPENAI_MODEL','gpt-6.1-sol'))
    p.add_argument('--api-key',help='Prefer OPENAI_API_KEY to avoid shell history/process-list exposure')
    p.add_argument('--base-url',default=os.environ.get('OPENAI_BASE_URL',DEFAULT_BASE_URL))
    p.add_argument('--output',type=Path,default=Path('outputs/streamdrivebench'))
    p.add_argument('--reasoning-effort',default='low',choices=['none','minimal','low','medium','high','xhigh'])
    p.add_argument('--workers',type=int,default=4,help='Max simultaneous video chains and API requests')
    p.add_argument('--max-events-per-window',type=int,default=1)
    p.add_argument('--max-calls-per-video',type=int,default=0,help='Per-run request attempt limit per video; 0=unlimited')
    p.add_argument('--limit-videos',type=int,help='First N paths; use a separate output for a different selection')
    p.add_argument('--dry-run',action='store_true',help='Count/probe videos without API, decoding or output writes')
    a=p.parse_args(argv)
    if min(a.workers,a.max_events_per_window)<1 or a.max_calls_per_video<0 or (a.limit_videos is not None and a.limit_videos<1):p.error('Invalid limits')
    parsed=urlsplit(a.base_url)
    if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:p.error('Base URL must be a plain HTTP(S) endpoint without credentials/query/fragment')
    if a.api_key:os.environ['OPENAI_API_KEY']=a.api_key
    source=a.video_dir.resolve();root=a.output.resolve()
    if root==source or root in source.parents or source in root.parents:p.error('Output and video directory must be independent trees')
    clips=discover_clips(source)
    if a.limit_videos is not None:clips=clips[:a.limit_videos]
    print(json.dumps(dict(selected_videos=len(clips),model=a.model,reasoning_effort=a.reasoning_effort,workers=a.workers,
        window_seconds=30,max_events_per_window=a.max_events_per_window,output=str(root))),flush=True)
    if a.dry_run:return
    if not os.environ.get('OPENAI_API_KEY'):p.error('Provide OPENAI_API_KEY or --api-key; no requests sent')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=dict(model=a.model,effort=a.reasoning_effort,base_url=a.base_url,max_output_tokens=6500)
        entries=[dict(clip=c,episode_id=c['episode_id'],raw=str(root/'videos'/c['episode_id']),source_sha256=file_hash(source/c['file'])) for c in clips]
        contract=dict(version='streamdrivebench-build-v1',video_root=str(source),entries=entries,config=config,
            max_events_per_window=a.max_events_per_window,max_calls_per_video=a.max_calls_per_video,
            code_hashes={f.name:file_hash(f) for f in Path(__file__).parent.glob('*.py')})
        freeze(root/'build_plan.json',contract)
        ready=[]
        for e in entries:
            raw=Path(e['raw'])
            if (raw/'status.json').exists() and load_json(raw/'status.json')['status']=='complete_pending_human_review':
                if not load_json(raw/'structural_audit.json')['passed']:raise ValueError('Completed video failed audit')
                ready.append(e)
        def update(done,active,failures):
            completed=ready+done
            summary=dict(status='failed' if failures and not active else 'draining_after_failure' if failures else 'running',
                total_videos=len(entries),completed_videos=len(completed),active_videos=[e['clip']['file'] for e in active],failures=failures)
            dump_json(summary,root/'status.json');print(json.dumps(summary,ensure_ascii=False),flush=True)
        def work(e):build(e['clip'],source,Path(e['raw']),root/'api_cache',config,a.max_events_per_window,a.max_calls_per_video)
        try:
            _,failures=execute([e for e in entries if e not in ready],a.workers,work,update)
            if failures:raise RuntimeError('Some videos failed; checkpoints retained. Re-run the same command after resolving the error; see status.json.')
            result=publish(entries,root/'release')
            dump_json(result,root/'summary.json');dump_json(dict(status='complete_pending_human_review',total_videos=len(entries),completed_videos=len(entries)),root/'status.json')
            print('FINAL_QA_COUNTS '+json.dumps(result['by_family'])+f' total={result["queries"]}',flush=True)
            print('RELEASE '+str(root/'release'),flush=True)
        finally:usage(root)
