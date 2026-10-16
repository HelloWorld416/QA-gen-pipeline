import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from streamdrivebench.release import balance,validate


def test_global_balance_and_silent_last_preserve_semantics():
    rows=[]
    for fam in ('HGI','AD','ETP'):
        for n in (3,4,5):
            for i in range(19):
                choices={chr(65+k):f'option{k}' for k in range(n)}
                negative=fam=='ETP' and i%5==0
                if fam=='ETP':choices['A']='SILENT'
                answer='A' if fam!='ETP' or negative else 'B'
                rows.append(dict(query_id=f'{fam}-{n}-{i}',family=fam,model_input=dict(question='same text',choices=choices),
                    evaluator_only=dict(ground_truth=dict(answer=answer,outcome='not_observed_within_window' if negative else 'positive'))))
    new,audit=balance(rows);validate(rows,new,audit)
    assert balance(rows)==(new,audit)
    for q in new:
        if q['family']=='ETP':assert list(q['model_input']['choices'].values())[-1]=='SILENT'
