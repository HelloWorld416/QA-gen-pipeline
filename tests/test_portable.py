from pathlib import Path
import json
from fractions import Fraction
import av
from PIL import Image
import pytest
from streamdrivebench import cli, pipeline
from streamdrivebench.media import discover_clips,prepare_episode
from streamdrivebench.common import file_hash,load_json,validate_schema


def video(path,frames=24,rate=10,offset=0):
    path.parent.mkdir(parents=True,exist_ok=True)
    with av.open(str(path),'w') as out:
        s=out.add_stream('mpeg4',rate=rate);s.width=64;s.height=48;s.pix_fmt='yuv420p'
        for i in range(frames):
            frame=av.VideoFrame.from_image(Image.new('RGB',(64,48),(i%255,20,30)))
            frame.pts=i+offset;frame.time_base=Fraction(1,rate)
            for packet in s.encode(frame):out.mux(packet)
        for packet in s.encode():out.mux(packet)


def test_nested_inventory_and_arbitrary_short_duration(tmp_path):
    root=tmp_path/'videos';video(root/'a/same.mp4');video(root/'b/same.MP4',frames=1)
    clips=discover_clips(root)
    assert len(clips)==2 and clips[0]['episode_id']!=clips[1]['episode_id']
    for c in clips:
        ep=prepare_episode(dict(c,sha256=file_hash(root/c['file'])),root,tmp_path/c['episode_id'])
        assert ep['num_frames']==(3 if c['file'].startswith('a/') else 1)
        assert all(f['source_video_time_sec']<=f['frame_index'] for f in ep['frames'])
        again=prepare_episode(dict(c,sha256=file_hash(root/c['file'])),root,tmp_path/c['episode_id'])
        assert again==ep


def test_nonzero_pts_origin_is_normalized(tmp_path):
    video(tmp_path/'in/v.mp4',offset=50)
    c=discover_clips(tmp_path/'in')[0]
    e=prepare_episode(dict(c,sha256=file_hash(tmp_path/'in/v.mp4')),tmp_path/'in',tmp_path/'out')
    assert e['source_pts_origin_sec']>0 and e['frames'][0]['source_video_time_sec']==0
    assert 2.3<e['duration_sec']<2.5


def test_cli_complete_empty_release_and_zero_api_resume(tmp_path,monkeypatch):
    video(tmp_path/'in/a.mp4');video(tmp_path/'in/sub/a.mp4')
    calls=[]
    class FakeClient:
        def __init__(self,*a,**k):pass
        def call(self,prompt,payload,images,schema):
            calls.append(payload);assert payload['stage']=='event_discovery'
            assert all(Path(i).exists() for i in images)
            value={'events':[],'coverage_notes':'No clear transition.'};validate_schema(value,schema);return value
    monkeypatch.setattr(pipeline,'Client',FakeClient);monkeypatch.setenv('OPENAI_API_KEY','offline-fixture')
    args=['--video-dir',str(tmp_path/'in'),'--model','fixture-model','--output',str(tmp_path/'out')]
    cli.main(args);assert len(calls)==2
    result=load_json(tmp_path/'out/summary.json');assert result['videos']==2 and result['by_family']=={'HGI':0,'AD':0,'ETP':0}
    assert load_json(tmp_path/'out/release/queries_all.json')==[]
    cli.main(args);assert len(calls)==2
    video(tmp_path/'in/extra.mp4')
    with pytest.raises(ValueError,match='Frozen'):cli.main(args)


def test_failures_stop_queue_and_keep_success(tmp_path):
    started=[]
    def work(e):
        started.append(e['clip']['file'])
        if len(started)==2:raise ValueError('fixture failure')
    entries=[{'clip':{'file':str(i)}} for i in range(5)]
    done,failed=cli.execute(entries,1,work,lambda *args:None)
    assert len(done)==1 and len(failed)==1 and started==['0','1']


def test_dry_run_no_key_and_no_output(tmp_path,monkeypatch):
    video(tmp_path/'in/a.mp4');monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    cli.main(['--video-dir',str(tmp_path/'in'),'--output',str(tmp_path/'out'),'--dry-run'])
    assert not (tmp_path/'out').exists()


def test_nonempty_end_to_end_public_labels_and_balanced_release(tmp_path,monkeypatch):
    from test_pipeline import Client as FixtureClient
    class FullFixtureClient(FixtureClient):
        def __init__(self,*args,**kwargs):super().__init__()
    video(tmp_path/'in/drive.mp4',frames=60,rate=1)
    monkeypatch.setattr(pipeline,'Client',FullFixtureClient);monkeypatch.setenv('OPENAI_API_KEY','offline-fixture')
    cli.main(['--video-dir',str(tmp_path/'in'),'--output',str(tmp_path/'out'),'--model','fixture'])
    r=tmp_path/'out/release';summary=load_json(r/'summary.json')
    assert summary['by_family']=={'HGI':1,'AD':1,'ETP':1}
    rows=load_json(r/'queries_all.json');public=load_json(r/'model_inputs.json');labels=load_json(r/'labels.json')
    mapping=load_json(r/'option_mapping_audit.json')
    for q,p in zip(rows,public):
        assert 'evaluator_only' not in p
        assert labels[q['query_id']]==q['evaluator_only']
        assert q['model_input']['choices'][labels[q['query_id']]['ground_truth']['answer']]=='a'
        assert mapping[q['query_id']]['letter_to_text']==q['model_input']['choices']
        if q['family']=='ETP':assert list(q['model_input']['choices'].values())[-1]=='SILENT'
    manifest=load_json(r/'release_manifest.json')
    for name,expected in manifest['files'].items():assert file_hash(r/name)==expected
