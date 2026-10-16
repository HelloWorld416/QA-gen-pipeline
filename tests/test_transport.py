import json
from io import BytesIO
from pathlib import Path
import threading
from streamdrivebench.client import Client
from streamdrivebench.common import obj_schema,STRING


def test_responses_contract_usage_and_cache(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','offline-fixture')
    class Data:
        image_size=[910,512];episode_by_id={'e':{'frames':[{'source_video_time_sec':0}]}}
        def valid(self,eid,frames):return frames==[0]
    config=dict(model='fixture',effort='low',base_url='https://example.test/v1',max_output_tokens=100)
    calls=[]
    def open_url(req,timeout):
        calls.append(req);body=json.loads(req.data)
        assert req.full_url=='https://example.test/v1/responses'
        assert body['reasoning']=={'effort':'low'} and body['text']['format']['strict']
        assert body['input'][0]['content'][2]['type']=='input_image'
        return BytesIO(json.dumps(dict(status='completed',output=[dict(content=[dict(type='output_text',text='{"value":"ok"}')])],usage={'input_tokens':100,'output_tokens':5,'total_tokens':105},id='fake')).encode())
    monkeypatch.setattr('urllib.request.urlopen',open_url)
    image=tmp_path/'frame.jpg';image.write_bytes(b'offline image fixture')
    client=Client(tmp_path,config,Data(),'e',threading.Semaphore(1),0)
    args=('prompt',dict(stage='fixture',image_frames=[0]),[image],obj_schema({'value':STRING}))
    assert client.call(*args)=={'value':'ok'} and client.call(*args)=={'value':'ok'} and len(calls)==1
    assert 'offline-fixture' not in ''.join(p.read_text() for p in tmp_path.rglob('*.json'))
