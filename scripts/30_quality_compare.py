"""Structural paired-output diagnostics, not a business accuracy evaluator."""
import argparse,collections,hashlib,json,pathlib

def load(path):
    records=[json.loads(s) for s in pathlib.Path(path).read_text().splitlines()]
    assert len(records)==150 and len({r['event_id'] for r in records})==150
    return {r['event_id']:r for r in records}

def reconstruct(r):
    text={k:'' for k in ['content','reasoning_content','reasoning']}
    calls={}
    assert 'output_deltas' in r or r['status']!='ok','Missing saved output'
    for d in r.get('output_deltas',[]):
        for k in text: text[k]+=d.get(k) or ''
        for c in d.get('tool_calls') or []:
            index=c.get('index',0)
            out=calls.setdefault(index,{'name':'','arguments':''})
            fn=c.get('function') or {}
            for k in out: out[k]+=fn.get(k) or ''
    text['tool_calls']=[calls[k] for k in sorted(calls)]
    return text

def audit(r):
    output=reconstruct(r)
    bad=0
    for c in output['tool_calls']:
        try: json.loads(c['arguments'])
        except (ValueError,TypeError): bad+=1
    body=''.join(output[k] for k in ['content','reasoning_content','reasoning'])
    repeated=False
    # Exact repeated suffix is a flag for inspection, not a semantic verdict.
    for width in [16,32,64,128,256]:
        if len(body)>=width*8 and body[-width:]*8==body[-width*8:]: repeated=True
    serialized=json.dumps(output,sort_keys=True,ensure_ascii=False)
    return {'status':r['status'],'empty':not body and not output['tool_calls'],
            'truncated':'length' in r.get('finish_reasons',[]),
            'tool_calls':len(output['tool_calls']),'malformed_tool_json':bad,
            'repeated_suffix_flag':repeated,'completion_tokens':r.get('completion_tokens'),
            'sha256':hashlib.sha256(serialized.encode()).hexdigest(),
            'finish_reasons':r.get('finish_reasons',[])}

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate');p.add_argument('--output',required=True);a=p.parse_args()
    base=load(a.baseline); groups={'baseline':base}
    if a.candidate:
        groups['candidate']=load(a.candidate)
        assert base.keys()==groups['candidate'].keys()
        for key in base:
            for field in ['index','scheduled_at_sec','diagnostic_temperature','diagnostic_seed']:
                assert base[key][field]==groups['candidate'][key][field],(key,field)
    result={'scope':'structural proxies only; no business labels; exact-output differences require baseline-repeat calibration','groups':{}}
    for name,records in groups.items():
        rows={key:audit(r) for key,r in records.items()}
        result['groups'][name]={'requests':len(rows),'success':sum(r['status']=='ok' for r in rows.values()),
            **{k:sum(r[k] for r in rows.values()) for k in ['empty','truncated','tool_calls','malformed_tool_json','repeated_suffix_flag']},'rows':rows}
    if a.candidate:
        b=result['groups']['baseline']['rows'];c=result['groups']['candidate']['rows']
        result['comparison']={'exact_outputs':sum(b[k]['sha256']==c[k]['sha256'] for k in b),
          'changed_event_ids':[k for k in b if b[k]['sha256']!=c[k]['sha256']],
          'new_structural_flags':{f:[k for k in b if c[k][f] and not b[k][f]] for f in ['empty','truncated','malformed_tool_json','repeated_suffix_flag']}}
    with pathlib.Path(a.output).open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:{field:value for field,value in v.items() if field!='rows'} for k,v in result['groups'].items()},ensure_ascii=False))
if __name__=='__main__':main()
