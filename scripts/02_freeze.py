import json,subprocess,hashlib,importlib.metadata as m,pathlib,datetime,os
r=pathlib.Path('/root/dsk_opt_goal_20261116/evidence')
x={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'packages':{d.metadata['Name']:d.version for d in m.distributions()},'processes':{},'files':{}}
for pid in [311625,317210]:
 p=pathlib.Path(f'/proc/{pid}'); x['processes'][str(pid)]={'cmdline':p.joinpath('cmdline').read_bytes().replace(bytes([0]),b' ').decode(),'pgid':os.getpgid(pid)}
base=pathlib.Path(m.distribution('vllm').locate_file('vllm'))
for f in ['model_executor/layers/fused_moe/experts/fused_humming_moe.py','models/deepseek_v41/nvidia/flashmla.py','model_executor/layers/sparse_attn_indexer.py']:
 p=base/f; x['files'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
r.joinpath('baseline-freeze.json').write_text(json.dumps(x,indent=2))
for cmd,name in [(['nvidia-smi'],'gpu-before.txt'),(['nvidia-smi','topo','-m'],'topology.txt'),(['ps','-eo','pid,ppid,pgid,comm'],'process-tree.txt')]: r.joinpath(name).write_text(subprocess.check_output(cmd,text=True))
print(json.dumps({'processes':x['processes'],'files':x['files']},indent=2))
