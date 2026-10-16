"""Discover and confirm events before shared-window QA authoring and verification."""
from collections import Counter, defaultdict
from pathlib import Path
import threading
from .common import dump_json, load_json, digest, file_hash, freeze
from .client import Client
from .media import Dataset, prepare_episode
from . import checks as P
from . import protocol as W

CAUSAL={'hgi_current','hgi_full','ad_prefix','injection','etp_commitment'}


class Runner:
    def __init__(self,root,episode,data,client,max_events=1):
        self.root=Path(root);self.ep=episode;self.eid=episode['episode_id'];self.n=episode['num_frames']
        self.data=data;self.client=client;self.max_events=max_events
        self.folder=self.root/'episodes'/self.eid;self.folder.mkdir(parents=True,exist_ok=True)
        path=self.folder/'request_audit.json'
        self.trace={r['request_id']:r for r in load_json(path)} if path.exists() else {}

    def call(self,prompt,payload,schema):
        frames=payload['image_frames'];stage=payload['stage']
        if not frames or not self.data.valid(self.eid,frames):raise ValueError('Invalid source evidence')
        if stage in CAUSAL:
            allowed={'stage','image_frames','query_second','family','horizon_seconds','question','injection_second'}
            if set(payload)-allowed or max(frames)>payload['query_second']:
                raise ValueError('Causal payload leakage')
        result=self.client.call(prompt,payload,[self.data.image_path(self.eid,t) for t in frames],schema)
        key=digest([prompt,payload,schema])
        self.trace[key]=dict(request_id=key,stage=stage,payload=payload,result=result)
        dump_json(list(self.trace.values()),self.folder/'request_audit.json')
        return result

    def causal(self,prompt,stage,frames,t,family,q,schema,horizon=0,**extra):
        payload=P.causal_payload(stage,frames,t,family,horizon,q);payload.update(extra)
        return self.call(prompt,payload,schema)

    def verify(self,candidate,start,end):
        family=candidate['family'];t=candidate['query_second'];h=candidate['horizon_seconds'];q=candidate['question']
        checks={};errors=W.preflight(candidate,start,end,self.n);label=None
        if errors:return checks,errors,label
        prefix=W.prefix(candidate.get('history_start_second',start),t)
        needed=prefix+(list(range(t,t+31)) if family=='ETP' else list(range(t,t+h+1)))
        if not self.data.valid(self.eid,needed):return checks,['source_gap'],label
        if family=='HGI':
            cur=self.causal(P.VERIFY_PROMPT,'hgi_current',[t],t,family,q,P.VERIFY)
            checks['current_only']=cur;errors+=P.evidence_errors(cur,[t])
            if cur['valid'] and cur['unambiguous']:errors.append('current_frame_sufficient')
            if errors:return checks,errors,label
            full=self.causal(P.VERIFY_PROMPT,'hgi_full',prefix,t,family,q,P.VERIFY)
            checks['full']=full;errors+=P.evidence_errors(full,prefix)
            if not all(full[k] for k in ('valid','target_identifiable','unambiguous','history_necessary')):errors.append('history_not_supported')
            if full['future_presupposition']:errors.append('future_presupposition')
            if not any(f<t for f in full['evidence_frames']):errors.append('missing_history_evidence')
            label=dict(answer=full['answer'],evidence_frames=full['evidence_frames'])
        elif family=='AD':
            pre=self.causal(P.AD_PREFIX_PROMPT,'ad_prefix',prefix,t,family,q,P.AD_PREFIX,h)
            checks['prefix']=pre;errors+=P.evidence_errors(pre,prefix)
            if not all(pre[k] for k in ('valid','target_identifiable','unambiguous')):errors.append('prefix_invalid')
            if pre['future_presupposition']:errors.append('future_presupposition')
            if errors:return checks,errors,label
            frames=list(range(max(candidate.get('history_start_second',start),t-2),t+h+1))
            fut=self.call(P.VERIFY_PROMPT,dict(stage='ad_future',image_frames=frames,query_second=t,
                horizon_seconds=h,family=family,question=clean_question(q)),P.VERIFY)
            checks['future']=fut;errors+=P.evidence_errors(fut,frames)
            if not all(fut[k] for k in ('valid','target_identifiable','unambiguous')):errors.append('future_invalid')
            if fut['future_presupposition']:errors.append('future_presupposition')
            if not any(f>t for f in fut['evidence_frames']):errors.append('no_future_evidence')
            label=dict(answer=fut['answer'],evidence_frames=fut['evidence_frames'],horizon_seconds=h)
        else:
            inj=self.causal(P.INJECTION_PROMPT,'injection',prefix,t,family,q,P.INJECTION)
            checks['injection']=inj
            if not all(inj[k] for k in ('valid','target_identifiable','single_atomic_condition','driving_state_change')) or inj['condition_already_true'] or inj['future_presupposition']:
                return checks,['injection_invalid'],label
            frames=list(range(t,t+31))
            mon=self.call(P.MONITOR_PROMPT,dict(stage='monitor',image_frames=frames,
                injection_second=t,deadline_second=t+30,question=clean_question(q)),P.MONITOR)
            checks['monitor']=mon;errors+=P.evidence_errors(mon,frames)+P.monitor_errors(mon,t,t+30,q['choices'])
            label=dict(answer=mon['answer'],evidence_frames=mon['evidence_frames'],trigger_second=mon['first_observable'],
                last_absent=mon['last_absent'],outcome='positive' if mon['first_observable']!=-1 else 'not_observed_within_window',
                monitor_end_second=t+30,response_grace_seconds=1)
        if label and label['answer'] not in q['choices']:errors.append('answer_not_a_choice')
        return checks,list(dict.fromkeys(errors)),label

    def commitment(self,row,start):
        t=row['query_second'];trigger=row['label']['trigger_second'];checks=[];errors=[]
        for cutoff in (trigger-1,trigger):
            frames=sorted(set(W.prefix(row.get('history_start_second',start),t)+list(range(t,cutoff+1))))
            value=self.causal(W.COMMIT_PROMPT,'etp_commitment',frames,cutoff,'ETP',row['question'],W.COMMIT,injection_second=t)
            checks.append(dict(cutoff=cutoff,result=value));errors+=P.evidence_errors(value,frames)
            if not value['target_identifiable'] or not value['evidence_decidable']:errors.append('causal_boundary_uncertain')
            if cutoff not in value['evidence_frames']:errors.append('causal_latest_evidence_missing')
        previous,current=[c['result'] for c in checks]
        if previous['may_answer_now'] or previous['answer']!='SILENT':errors.append('causal_trigger_already_answerable_earlier')
        if not current['may_answer_now'] or current['answer']!=row['label']['answer']:errors.append('causal_trigger_not_answerable_now')
        row['checks']['commitment']=checks
        if not errors:row['label']['causal_confirmation_checked']=True
        return errors

    def window(self,start,end,requested_families=None):
        requested_families=list(requested_families if requested_families is not None else ('HGI','AD','ETP'))
        if not requested_families:raise ValueError('No requested families')
        frames=list(range(start,end));events=[];candidates=[];accepted=[];rejections=[]
        if not self.data.valid(self.eid,frames):
            return dict(start=start,end_exclusive=end,events=[],candidates=[],accepted=[],rejections=[{'reason':['source_gap_window']}])
        proposal=self.call(W.DISCOVER_PROMPT,dict(stage='event_discovery',image_frames=frames,
            ownership_start=start,ownership_end_exclusive=end,max_events=self.max_events),W.PROPOSAL)
        if len(proposal['events'])>self.max_events:raise ValueError('Proposal exceeded event quota')
        seen=set();event_keys=set()
        for i,proposed in enumerate(proposal['events']):
            annotation=proposed
            if not start<annotation['approximate_transition']<end or not set(annotation['evidence_frames']).issubset(frames):
                rejections.append(dict(reason=['proposal_outside_window']));continue
            boundary=self.call(P.REFINE_PROMPT,dict(stage='window_boundary',image_frames=frames,proposal=annotation),P.BOUNDARY)
            t=boundary['first_observable']
            if not boundary['valid'] or not boundary['visibility_clear'] or not start<t<end or boundary['last_absent']!=t-1 or not {t-1,t}.issubset(boundary['evidence_frames']) or P.evidence_errors(boundary,frames):
                rejections.append(dict(reason=['uncertain_event_boundary']));continue
            event_key=(boundary['actor'].strip().casefold(),t)
            if event_key in event_keys:
                rejections.append(dict(reason=['duplicate_confirmed_transition']));continue
            event_keys.add(event_key)
            event_id=f'{self.eid}-w{start:06d}-e{i+1:02d}'
            event=dict(event_id=event_id,transition_id=event_id,episode_id=self.eid,
                source_sequence=self.ep['source_sequence'],discovery_start=start,discovery_end_exclusive=end,
                transition_second=t,actor=boundary['actor'],behavior=boundary['behavior'],
                before_state=boundary['before_state'],after_state=boundary['after_state'],
                evidence_frames=boundary['evidence_frames'],boundary_check=boundary,annotation_source='window30_confirmed_before_author')
            events.append(event)
        # No QA request until ALL event candidates in this window are reviewed.
        # Empty/rejected windows therefore spend zero author/QA verification calls.
        if not events:
            return dict(start=start,end_exclusive=end,events=[],candidates=[],accepted=[],rejections=rejections,authoring=[])
        authored=self.call(W.AUTHOR_PROMPT,dict(stage='qa_author',image_frames=frames,
            ownership_start=start,ownership_end_exclusive=end,video_num_seconds=self.n,
            confirmed_events=events,requested_families=requested_families),W.AUTHOR)['question_sets']
        ids=[s['event_id'] for s in authored]
        if len(ids)!=len(set(ids)) or set(ids)!={e['event_id'] for e in events}:
            raise ValueError('QA author must cover exactly the confirmed event IDs')
        by_id={s['event_id']:s for s in authored}
        for event in events:
            event_id=event['event_id'];t=event['transition_second'];boundary=event['boundary_check']
            authored_event=by_id[event_id]
            families=[q['family'] for q in authored_event['questions']]
            omissions=authored_event['omissions'];accounted=families+[o['family'] for o in omissions]
            if len(accounted)!=len(requested_families) or set(accounted)!=set(requested_families) or any(not o['reason'].strip() for o in omissions):
                raise ValueError('Each QA family requires one question or one justified omission')
            for candidate in authored_event['questions']:
                family=candidate['family'];q=candidate['question'];query=candidate['query_second']
                row=dict(query_id=event_id+'-'+family.lower(),event_id=event_id,transition_id=event_id,
                    episode_id=self.eid,family=family,query_second=query,history_start_second=candidate.get('history_start_second',start),question=q,checks={},errors=[],label=None,
                    strict_transition_link=False,probe_series_id=digest([event_id,family,q['predicate']])[:16],human_approved=False,
                    discovery_start=start,discovery_end_exclusive=end,authoring_policy='confirmed_event_then_offline_author_causal_audit')
                key=P.model_input_key(family,query,q)
                if key in seen:row['errors']=['duplicate_model_input']
                elif family in ('AD','ETP') and query>=t:
                    row['errors']=['query_not_before_confirmed_transition']
                else:
                    seen.add(key);row['checks'],row['errors'],row['label']=self.verify(candidate,start,end)
                if not row['errors']:
                    link_frames=sorted(set(boundary['evidence_frames']+row['label']['evidence_frames']+W.prefix(row['history_start_second'],query)))
                    # Include all endpoint evidence for AD and full monitoring for ETP.
                    if family=='AD':link_frames=sorted(set(link_frames+list(range(query,query+4))))
                    if family=='ETP':link_frames=sorted(set(link_frames+list(range(query,query+31))))
                    association=self.call(P.LINK_PROMPT,dict(stage='association',image_frames=link_frames,
                        event=event,family=family,query_second=query,horizon_seconds=candidate['horizon_seconds'],
                        question=clean_question(q),label=row['label'],previous_questions=[]),P.LINK)
                    row['association']=association;row['errors']+=P.association_errors(association)
                    row['strict_transition_link']=association['same_atomic_transition'] and (
                        family!='ETP' or (row['label']['outcome']=='positive' and row['label']['trigger_second']==t))
                if not row['errors'] and family=='ETP' and row['label']['outcome']=='positive':
                    row['errors']+=self.commitment(row,start)
                row['errors']=list(dict.fromkeys(row['errors']));candidates.append(row)
                if not row['errors']:accepted.append(row)
        return dict(start=start,end_exclusive=end,events=events,candidates=candidates,accepted=accepted,rejections=rejections,authoring=authored)

    def run(self):
        outputs=[]
        for start,end in W.windows(self.n):
            path=self.folder/'windows'/f'{start:06d}.json'
            value=load_json(path) if path.exists() else self.window(start,end)
            if (value['start'],value['end_exclusive'])!=(start,end):raise ValueError('Window checkpoint mismatch')
            if not path.exists():dump_json(value,path)
            outputs.append(value)
            status=dict(status='running',episode=self.eid,seconds_completed=end,total_seconds=self.n,
                completed_windows=len(outputs),accepted=sum(len(x['accepted']) for x in outputs))
            dump_json(status,self.root/'status.json');print(status,flush=True)
        return outputs


def clean_question(q):
    return {k:q[k] for k in ('question','choices','target_description','predicate','answer_slot')}


def export(root,episode,windows,trace,allow_sparse=False):
    root=Path(root);events=[e for w in windows for e in w['events']]
    candidates=[q for w in windows for q in w['candidates']];accepted=[q for w in windows for q in w['accepted']]
    coverage=[s for w in windows for s in range(w['start'],w['end_exclusive'])]
    if len(coverage)!=len(set(coverage)) or any(s<0 or s>=episode['num_frames'] for s in coverage):raise ValueError('Invalid/overlapping discovery windows')
    if not allow_sparse and coverage!=list(range(episode['num_frames'])):raise ValueError('Overlapping or missing discovery windows')
    flat=[];groups={e['event_id']:dict(event_id=e['event_id'],episode_id=episode['episode_id'],
        split_group=episode['source_sequence'],transitions=[e['transition_id']],qa_ids=[],families=[]) for e in events}
    for row in accepted:
        if row['errors']:raise ValueError('Accepted invalid QA')
        group=groups[row['event_id']];group['qa_ids'].append(row['query_id']);group['families'].append(row['family'])
        t=row['query_second'];family=row['family'];q=row['question']
        flat.append(dict(query_id=row['query_id'],episode_id=episode['episode_id'],family=family,query_second=t,
            model_input=dict(question=q['question'],choices=q['choices'],target_description=q['target_description'],predicate=q['predicate']),
            input_policy=dict(prefix_end_second=t,history_start_second=row.get('history_start_second',row['discovery_start']),
                online_monitor_end_second=t+30 if family=='ETP' else t,evaluation_session='independent_per_question'),
            evaluator_only=dict(event_id=row['event_id'],transition_id=row['transition_id'],probe_series_id=row['probe_series_id'],
                ground_truth=row['label'],strict_transition_link=row['strict_transition_link'],human_approved=False,
                source_sequence=episode['source_sequence'],authoring_policy=row['authoring_policy'],
                discovery_start=row['discovery_start'],discovery_end_exclusive=row['discovery_end_exclusive'])))
    strict=defaultdict(set)
    for q in accepted:
        if q['strict_transition_link']:strict[q['transition_id']].add(q['family'])
    strict_ids={k for k,v in strict.items() if v=={'HGI','AD','ETP'}}
    commitment=[q for q in candidates if 'commitment' in q['checks']]
    passed=sum(q['family']=='ETP' and q['label'].get('causal_confirmation_checked',False) for q in accepted)
    report=dict(status='complete_pending_human_review',episodes=1,discovery_windows=len(windows),
        discovery_policy='sparse_selected_windows' if allow_sparse else 'all_windows',
        selected_discovery_seconds=len(coverage),video_num_seconds=episode['num_frames'],
        verified_transitions=len(events),unique_events=len(events),qa_candidates=len(candidates),accepted_qa=len(accepted),
        accepted_by_family=dict(Counter(q['family'] for q in accepted)),etp_outcomes=dict(Counter(q['label']['outcome'] for q in accepted if q['family']=='ETP')),
        events_with_all_three=sum(set(g['families'])=={'HGI','AD','ETP'} for g in groups.values()),strict_triplet_transitions=sorted(strict_ids),
        rejection_reasons=dict(Counter(e for q in candidates for e in q['errors'])),
        event_rejection_reasons=dict(Counter(reason for w in windows for r in w['rejections'] for reason in r['reason'])),
        author_omissions=[dict(event_id=s['event_id'],**o) for w in windows for s in w.get('authoring',[]) for o in s['omissions']],
        causal_commitment_audit=dict(positive_candidates=len(commitment),passed=passed,rejected=len(commitment)-passed),
        limitations=['Offline full-window authorship is outcome-aware; causal audits do not eliminate selection/wording bias.',
                     'At most one event per window by default; not exhaustive discovery. Window boundaries can censor events.',
                     'ETP monitors a full 30 seconds across discovery boundaries; tail windows are never shortened into negatives.',
                     'History is selected per question within the discovery window, up to 30 seconds; evaluators must honor input_policy.history_start_second.',
                     'Automated checks are not human ground truth. Legacy reused releases have separate provenance.'])
    for group in groups.values():group['families']=sorted(set(group['families']))
    for name,value in dict(events=events,event_groups=list(groups.values()),qa_candidates=candidates,accepted_qa=accepted,
            queries_all=flat,model_inputs=[{k:v for k,v in q.items() if k!='evaluator_only'} for q in flat],
            queries_strict_triplets=[q for q in flat if q['evaluator_only']['transition_id'] in strict_ids and q['evaluator_only']['strict_transition_link']],
            report=report).items():dump_json(value,root/(name+'.json'))
    causal=0
    for call in trace:
        payload=call['payload'];frames=payload['image_frames']
        if call['stage'] in ('event_discovery','qa_author') and frames!=list(range(payload['ownership_start'],payload['ownership_end_exclusive'])):
            raise ValueError('Discovery context overlaps windows')
        if call['stage'] in CAUSAL:
            causal+=1
            if max(frames)>payload['query_second'] or set(payload)-{'stage','image_frames','query_second','family','horizon_seconds','question','injection_second'}:
                raise ValueError('Causal audit failed')
    dump_json(dict(passed=True,errors=[],discovery_seconds=len(coverage),causal_requests_checked=causal,
                   accepted_qa_checked=len(accepted),note='Structural audit; no real-video quality claim.'),root/'structural_audit.json')
    dump_json(dict(status=report['status']),root/'status.json')
    return report


def build(clip,video_root,root,cache,config,max_events=1,max_calls=0):
    root=Path(root);cache=Path(cache);root.mkdir(parents=True,exist_ok=True)
    source_hash=file_hash(Path(video_root)/clip['file'])
    dependencies=list(Path(__file__).parent.glob('*.py'))
    contract=dict(version='streamdrivebench-events-v1',config=dict(**config,max_events_per_window=max_events,
        prepared_root=str(root/'source_1fps'),cache_root=str(cache),max_calls_per_video=max_calls),
        clip=clip,video_sha256=source_hash,code_hashes={p.name:file_hash(p) for p in dependencies})
    freeze(root/'plan.json',contract)
    episode=prepare_episode(dict(clip,sha256=source_hash),Path(video_root),root)
    dump_json(dict(episodes=[episode]),root/'episodes_metadata.json')
    data=Dataset(root/'source_1fps',[episode]);client=Client(cache/'calls'/clip['episode_id'],config,data,clip['episode_id'],threading.Semaphore(1),max_calls)
    runner=Runner(root,episode,data,client,max_events)
    try:
        values=runner.run();report=export(root,episode,values,list(runner.trace.values()))
        usage=Counter();attempts=Counter()
        for path in (cache/'calls'/clip['episode_id']/'attempts').glob('*.json'):
            record=load_json(path);attempts[record['status']]+=1
            usage.update({k:v for k,v in record.get('usage',{}).items() if type(v) in (int,float)})
        dump_json(dict(usage=dict(usage),attempts=dict(attempts)),root/'token_usage.json')
        return report
    except Exception as exc:
        dump_json(dict(status='failed',error_type=type(exc).__name__),root/'status.json');raise
