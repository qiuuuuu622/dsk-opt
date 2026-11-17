"""Structural paired-output diagnostics, not a business accuracy evaluator."""
import argparse,collections,hashlib,json,pathlib,re,gzip

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

def dsml_check(text, allowed):
    # Spaced dialect from vLLM 0.32.0 parser/deepseek_v41.py.
    starts=text.count('<｜DSML｜ calls>');ends=text.count('</｜DSML｜ calls>')
    body=text.split('</think>',1)[-1]
    invoke=re.compile(r'<｜DSML｜ invoke name="([^"]+)">(.*?)</｜DSML｜ invoke>',re.S)
    param=re.compile(r'<｜DSML｜ parameter\s+name="([^"]+)"(?:\s+string="([^"]*)")?\s*>(.*?)</｜DSML｜ parameter>',re.S)
    calls=list(invoke.finditer(body)); malformed=int(starts!=ends)
    malformed+=int(body.count('<｜DSML｜ invoke ')!=len(calls) or body.count('</｜DSML｜ invoke>')!=len(calls))
    raw_json_failures=0;unknown=0;duplicates=0
    for call in calls:
        name,args=call.groups();unknown+=int(allowed is not None and name not in allowed)
        params=list(param.finditer(args));names=[p.group(1) for p in params]
        duplicates+=len(names)-len(set(names))
        malformed+=int(args.count('<｜DSML｜ parameter')!=len(params) or args.count('</｜DSML｜ parameter>')!=len(params))
        for p in params:
            if p.group(2)!='true':
                try:json.loads(p.group(3))
                except ValueError:raw_json_failures+=1
    return {'dsml_calls':len(calls),'dsml_unbalanced_flags':malformed,'dsml_unknown_tools':unknown,'dsml_duplicate_parameters':duplicates,'dsml_nonstring_json_fallbacks':raw_json_failures}

def audit(r, allowed=None):
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
    return {**dsml_check(body,allowed),'status':r['status'],'empty':not body and not output['tool_calls'],
            'truncated':'length' in r.get('finish_reasons',[]),
            'tool_calls':len(output['tool_calls']),'malformed_tool_json':bad,
            'repeated_suffix_flag':repeated,'completion_tokens':r.get('completion_tokens'),
            'sha256':hashlib.sha256(serialized.encode()).hexdigest(),
            'finish_reasons':r.get('finish_reasons',[])}

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate');p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    base=load(a.baseline); groups={'baseline':base}
    with gzip.open(a.plan,'rt') as f:plan=json.load(f)
    allowed={}
    for key,req in plan['resolved_requests'].items():
        allowed[key]={t['function']['name'] for t in plan['config_registry'][req['config_id']].get('tools',[]) if t.get('type')=='function'}
    if a.candidate:
        groups['candidate']=load(a.candidate)
        assert base.keys()==groups['candidate'].keys()
        for key in base:
            for field in ['index','scheduled_at_sec','diagnostic_temperature','diagnostic_seed']:
                assert base[key][field]==groups['candidate'][key][field],(key,field)
    result={'scope':'structural proxies only; no business labels; exact-output differences require baseline-repeat calibration','groups':{}}
    for name,records in groups.items():
        rows={key:audit(r,allowed[key]) for key,r in records.items()}
        result['groups'][name]={'requests':len(rows),'success':sum(r['status']=='ok' for r in rows.values()),
            **{k:sum(r[k] for r in rows.values()) for k in ['empty','truncated','tool_calls','malformed_tool_json','repeated_suffix_flag','dsml_calls','dsml_unbalanced_flags','dsml_unknown_tools','dsml_duplicate_parameters','dsml_nonstring_json_fallbacks']},'rows':rows}
    if a.candidate:
        b=result['groups']['baseline']['rows'];c=result['groups']['candidate']['rows']
        result['comparison']={'exact_outputs':sum(b[k]['sha256']==c[k]['sha256'] for k in b),
          'changed_event_ids':[k for k in b if b[k]['sha256']!=c[k]['sha256']],
          'new_structural_flags':{f:[k for k in b if c[k][f] and not b[k][f]] for f in ['empty','truncated','malformed_tool_json','repeated_suffix_flag','dsml_unbalanced_flags','dsml_unknown_tools','dsml_duplicate_parameters']}}
    with pathlib.Path(a.output).open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:{field:value for field,value in v.items() if field!='rows'} for k,v in result['groups'].items()},ensure_ascii=False))
if __name__=='__main__':main()
