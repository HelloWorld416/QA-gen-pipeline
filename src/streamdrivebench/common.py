import hashlib
import json
import os
from pathlib import Path
import tempfile

STRING={'type':'string'}
INTEGER={'type':'integer'}
BOOL={'type':'boolean'}
INTS={'type':'array','items':INTEGER}
STRINGS={'type':'array','items':STRING}


def obj_schema(properties):
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def dump_json(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as f:
        tmp=Path(f.name)
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    try:os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)


def freeze(path,value):
    path=Path(path)
    if path.exists() and load_json(path)!=value:
        raise ValueError(f'Frozen configuration/source changed: {path}; use a new output directory')
    dump_json(value,path)


def validate_schema(value,schema):
    kind=schema.get('type')
    valid={'object':type(value) is dict,'array':type(value) is list,'integer':type(value) is int,
           'number':type(value) in (int,float),'boolean':type(value) is bool,'string':type(value) is str}
    if kind and not valid.get(kind,False):raise ValueError('Response schema type mismatch')
    if 'enum' in schema and value not in schema['enum']:raise ValueError('Response enum mismatch')
    if kind=='object':
        if set(schema.get('required',[]))-set(value):raise ValueError('Missing response keys')
        if schema.get('additionalProperties') is False and set(value)-set(schema['properties']):raise ValueError('Unexpected response keys')
        for k,v in value.items():
            if k in schema['properties']:validate_schema(v,schema['properties'][k])
    if kind=='array':
        if len(value)<schema.get('minItems',0) or len(value)>schema.get('maxItems',float('inf')):raise ValueError('Response array size mismatch')
        for item in value:validate_schema(item,schema['items'])
    if kind in ('integer','number'):
        if value<schema.get('minimum',-float('inf')) or value>schema.get('maximum',float('inf')):raise ValueError('Response number out of range')
