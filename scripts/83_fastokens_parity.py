import os,json,gzip,pathlib,hashlib,time,importlib.metadata as md
from vllm.tokenizers import get_tokenizer
r=pathlib.Path('/root/dsk_opt_goal_20261116');out=r/'evidence/fastokens-parity.json'
os.environ['VLLM_USE_FASTOKENS']='0';native=get_tokenizer('/model',tokenizer_mode='deepseek_v41')
os.environ['VLLM_USE_FASTOKENS']='1';fast=get_tokenizer('/model',tokenizer_mode='deepseek_v41')
assert type(fast._tokenizer).__module__.startswith('fastokens'),type(fast._tokenizer)
assert not type(native._tokenizer).__module__.startswith('fastokens')
with gzip.open('/root/bench_dsk_first150_20261116/first150.plan.json.gz','rt') as f:p=json.load(f)
rows=[]
for idx,event in enumerate(p['schedule']):
 req=p['resolved_requests'][event['event_id']];cfg=p['config_registry'][req['config_id']]
 kw=dict(cfg.get('chat_template_kwargs') or {});kw['tools']=cfg.get('tools')
 prompt=native.apply_chat_template(req['full_messages'],tokenize=False,**kw)
 assert prompt==fast.apply_chat_template(req['full_messages'],tokenize=False,**kw)
 t=time.perf_counter();a=native.encode(prompt,add_special_tokens=False);ta=time.perf_counter()-t
 t=time.perf_counter();b=fast.encode(prompt,add_special_tokens=False);tb=time.perf_counter()-t
 assert a==b,('token mismatch',idx)
 for skip in [False,True]:assert native.decode(a,skip_special_tokens=skip)==fast.decode(b,skip_special_tokens=skip),('decode mismatch',idx,skip)
 rows.append({'index':idx+1,'tokens':len(a),'ids_sha256':hashlib.sha256(json.dumps(a).encode()).hexdigest(),'native_seconds':ta,'fast_seconds':tb})
 print('parity',idx+1,flush=True)
out.write_text(json.dumps({'passed':True,'version':md.version('fastokens'),'native_type':str(type(native._tokenizer)),'fast_type':str(type(fast._tokenizer)),'scope':'150 original chats rendered with tools and chat_template_kwargs; prompt, token IDs, decoded text exact; image placeholders not vision embeddings','rows':rows},indent=2))
