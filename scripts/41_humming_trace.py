"""Post-benchmark kernel evidence only; requests excluded from timing results."""
import argparse,asyncio,collections,gzip,json,pathlib,time,urllib.request
p=argparse.ArgumentParser();p.add_argument('--group',required=True);a=p.parse_args()
root=pathlib.Path('/root/dsk_opt_goal_20261116');g=root/'groups'/a.group
assert json.loads((g/'evidence/run-state.json').read_text())['status']=='completed_service_retained'
assert not (g/'evidence/profile-diagnostic.json').exists()
control=[]
def post(name):
 req=urllib.request.Request('http://127.0.0.1:8081/engine/control/'+name,data=json.dumps({'profile_prefix':a.group+'-kernel-proof'}).encode(),headers={'Content-Type':'application/json'},method='POST')
 with urllib.request.urlopen(req,timeout=180) as response:result=json.load(response)
 control.append({'operation':name,'response':result});return result
with gzip.open('/root/bench_dsk_first150_20261116/first150.plan.json.gz','rt') as f:plan=json.load(f)
async def replay():
 from openai import AsyncOpenAI
 async with AsyncOpenAI(base_url='http://127.0.0.1:8000/v1',api_key='TEST_ONLY_UNUSED',timeout=180,max_retries=0) as client:
  async def one(event):
   req=plan['resolved_requests'][event['event_id']]
   payload={k:v for k,v in plan['config_registry'][req['config_id']].items() if v is not None}
   payload.update(model='sf-finance-dsk41-dynamo-vllm-alyeast-prod',messages=req['full_messages'],max_tokens=256,stream=False)
   payload.pop('stream_options',None)
   response=await client.chat.completions.create(**payload)
   return {'event_id':event['event_id'],'usage':response.usage.model_dump() if response.usage else None,'finish_reasons':[x.finish_reason for x in response.choices]}
  return await asyncio.gather(*(one(e) for e in plan['schedule'][:3]))
started=post('start_profile');assert started.get('status')=='ok',started
try:records=asyncio.run(replay())
finally:post('stop_profile')
report={'purpose':'kernel path proof; not performance or quality metrics','requests':records,'control':control,'limit_tokens_per_request':256,'count':3,'profiles':[]}
for attempt in range(60):
 files=list((g/'traces').rglob('*.json.gz'))
 if len(files)>=8:
  try:
   profiles=[]
   for file in files:
    with gzip.open(file,'rt') as f:data=json.load(f)
    kernels=collections.defaultdict(lambda:{'count':0,'duration_us':0.0})
    for event in data.get('traceEvents',[]):
     if event.get('cat')=='kernel':
      row=kernels[event['name']];row['count']+=1;row['duration_us']+=event.get('dur',0)
    profiles.append({'file':str(file),'gpu_kernel_events':sum(x['count'] for x in kernels.values()),'kernels':dict(sorted(kernels.items(),key=lambda x:-x[1]['duration_us']))})
   if sum(x['gpu_kernel_events']>0 for x in profiles)>=8:
    report['profiles']=profiles;break
  except (EOFError,json.JSONDecodeError,OSError):pass
 time.sleep(1)
else:raise RuntimeError('Expected at least 8 complete GPU traces')
(g/'evidence/profile-diagnostic.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'group':a.group,'requests':len(records),'traces':len(report['profiles']),'GPU_events':[x['gpu_kernel_events'] for x in report['profiles']]}))
