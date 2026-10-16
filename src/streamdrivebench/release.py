"""Global option balance with SILENT last; all labels remain private."""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import random
from .common import dump_json, load_json, file_hash

def balance(rows,seed=20261015):
    groups=defaultdict(list);result={};audit={}
    for q in rows:
        cs=q['model_input']['choices'];gt=q['evaluator_only']['ground_truth'];answer=gt['answer']
        if q['query_id'] in result:raise ValueError('Duplicate query ID')
        result[q['query_id']]=None
        if not 2<=len(cs)<=5 or answer not in cs or len(set(cs.values()))!=len(cs):raise ValueError('Invalid choices/answer')
        silent=[k for k,v in cs.items() if v=='SILENT']
        if q['family']=='ETP':
            if len(silent)!=1:raise ValueError('ETP requires one SILENT')
            if (gt['outcome']=='not_observed_within_window')!=(answer==silent[0]):raise ValueError('ETP outcome/answer conflict')
        elif silent:raise ValueError('SILENT in non-ETP')
        groups[q['family'],len(cs),cs[answer]=='SILENT'].append(q)
    for (family,n,is_silent),items in sorted(groups.items()):
        rng=random.Random(f'{seed}:{family}:{n}:{is_silent}')
        items=sorted(items,key=lambda q:q['query_id']);rng.shuffle(items)
        width=n-1 if family=='ETP' else n
        slots=[]
        while len(slots)<len(items):
            block=list(range(width));rng.shuffle(block);slots.extend(block)
        for q,slot in zip(items,slots):
            old=q['model_input']['choices'];old_answer=q['evaluator_only']['ground_truth']['answer'];text=old[old_answer]
            if family=='ETP':
                values=[v for v in old.values() if v not in (text,'SILENT')];rng.shuffle(values)
                if not is_silent:values.insert(slot,text)
                values.append('SILENT')
            else:
                values=[v for v in old.values() if v!=text];rng.shuffle(values);values.insert(slot,text)
            choices={chr(65+i):v for i,v in enumerate(values)}
            new_answer=next(k for k,v in choices.items() if v==text)
            new=deepcopy(q);new['model_input']['choices']=choices;new['evaluator_only']['ground_truth']['answer']=new_answer
            result[q['query_id']]=new
            audit[q['query_id']]=dict(original_choices=list(old.values()),original_answer=text,
                source_letter_to_text=old,source_answer=old_answer,old_to_new={k:next(a for a,b in choices.items() if b==v) for k,v in old.items()},
                letter_to_text=choices,answer=new_answer)
    return [result[q['query_id']] for q in rows],audit


def validate(old,new,audit):
    groups=defaultdict(Counter)
    assert len(old)==len(new)==len(audit)
    for before,after in zip(old,new):
        cs=after['model_input']['choices'];a=after['evaluator_only']['ground_truth']['answer']
        oldcs=before['model_input']['choices'];olda=before['evaluator_only']['ground_truth']['answer']
        assert sorted(cs.values())==sorted(oldcs.values()) and cs[a]==oldcs[olda]
        restored=deepcopy(after);restored['model_input']['choices']=oldcs;restored['evaluator_only']['ground_truth']['answer']=olda
        assert restored==before
        fam=after['family'];n=len(cs)
        if fam=='ETP':assert list(cs.values())[-1]=='SILENT'
        if cs[a]!='SILENT':groups[fam,n][a]+=1
        assert audit[after['query_id']]['letter_to_text']==cs
    for (fam,n),counts in groups.items():
        valid='ABCDE'[:n-1 if fam=='ETP' else n];values=[counts[k] for k in valid]
        assert max(values)-min(values)<=1,(fam,n,counts)



def publish(entries, output):
    """Publish all completed videos together, never silently publish partial totals."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];episodes=[];strict=set();events=[];groups=[];sources=[];counts=[]
    for entry in entries:
        root=Path(entry['raw']);eid=entry['episode_id']
        if load_json(root/'status.json')['status']!='complete_pending_human_review':raise ValueError('Incomplete source')
        if not load_json(root/'structural_audit.json')['passed']:raise ValueError('Failed source audit')
        batch=load_json(root/'queries_all.json');rows.extend(batch)
        episodes.extend(load_json(root/'episodes_metadata.json')['episodes'])
        strict.update(q['query_id'] for q in load_json(root/'queries_strict_triplets.json'))
        events.extend(load_json(root/'events.json'));groups.extend(load_json(root/'event_groups.json'))
        media=(root/'source_1fps/frames'/eid).resolve();link=output/'source_1fps/frames'/eid
        link.parent.mkdir(parents=True,exist_ok=True)
        if link.exists() or link.is_symlink():
            if link.resolve()!=media:raise ValueError('Media link conflict')
        else:link.symlink_to(media,target_is_directory=True)
        sources.append(dict(episode_id=eid,raw=str(root),queries_sha256=file_hash(root/'queries_all.json')))
        counts.append(dict(video=entry['clip']['file'],queries=len(batch),by_family=dict(Counter(q['family'] for q in batch))))
    # Convert generated text choices to temporary letter inputs, then balance globally.
    source_rows=[]
    for q in rows:
        r=deepcopy(q);choices=q['model_input']['choices'];answer=q['evaluator_only']['ground_truth']['answer']
        r['model_input']['choices']={chr(65+i):v for i,v in enumerate(choices)}
        r['evaluator_only']['ground_truth']['answer']=chr(65+choices.index(answer));source_rows.append(r)
    balanced,mapping=balance(source_rows);validate(source_rows,balanced,mapping)
    totals={f:sum(q['family']==f for q in balanced) for f in ('HGI','AD','ETP')}
    summary=dict(status='complete_pending_human_review',videos=len(entries),queries=len(balanced),by_family=totals,per_video=counts,
        etp_outcomes=dict(Counter(q['evaluator_only']['ground_truth']['outcome'] for q in balanced if q['family']=='ETP')))
    objects=dict(queries_all=balanced,model_inputs=[{k:v for k,v in q.items() if k!='evaluator_only'} for q in balanced],
        labels={q['query_id']:q['evaluator_only'] for q in balanced},option_mapping_audit=mapping,
        queries_strict_triplets=[q for q in balanced if q['query_id'] in strict],episodes_metadata={'episodes':episodes},
        events=events,event_groups=groups,source_releases=sources,summary=summary,
        media={'prepared_root':str((output/'source_1fps').resolve())})
    for name,value in objects.items():dump_json(value,output/(name+'.json'))
    manifest=dict(version='streamdrivebench-balanced-silent-last-v1',queries=len(balanced),episodes=len(entries),by_family=totals,
        human_approved=False,files={name+'.json':file_hash(output/(name+'.json')) for name in objects})
    dump_json(manifest,output/'release_manifest.json')
    return summary
