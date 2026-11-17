import asyncio,json,pathlib,time
from openai import AsyncOpenAI
async def main():
 prefix=f'cache-probe-{time.time_ns()}'
 client=AsyncOpenAI(base_url='http://127.0.0.1:8000/v1',api_key='TEST_ONLY_UNUSED',max_retries=0,timeout=120)
 rows=[]
 try:
  for suffix in ['a','a','b','b']:
   salt=prefix+'-'+suffix
   response=await client.chat.completions.create(model='sf-finance-dsk41-dynamo-vllm-alyeast-prod',messages=[{'role':'user','content':('This is a cache isolation diagnostic. '*300)+'Return OK.'}],max_tokens=1,temperature=0,seed=1234,extra_body={'cache_salt':salt,'nvext':{'cache_salt':salt}})
   rows.append({'salt':suffix,'usage':response.usage.model_dump() if response.usage else None,'finish_reasons':[x.finish_reason for x in response.choices]})
 finally: await client.close()
 path=pathlib.Path('/root/dsk_opt_goal_20261116/evidence/cache-salt-probe.json');path.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
asyncio.run(main())
