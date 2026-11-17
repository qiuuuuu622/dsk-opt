import pathlib,json,os,signal,subprocess,time
r=pathlib.Path('/root/dsk_opt_goal_20261116');g=r/'groups/P-E1-FAST-HUM-ALL-r1'
pids=json.loads((g/'evidence/service-pids.json').read_text())
assert json.loads((g/'evidence/replay-exit.json').read_text())['exit_code']==0
for key in ['worker','frontend']:
 pid=pids[key];p=pathlib.Path(f'/proc/{pid}')
 if not p.exists():continue
 env=dict(x.split('=',1) for x in (p/'environ').read_bytes().decode().split('\0') if '=' in x)
 assert env.get('DYN_NAMESPACE')=='p-e1-fast-hum-all-r1';assert os.getpgid(pid)==pid;os.killpg(pid,signal.SIGTERM)
for _ in range(36):
 try:
  out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10)
  if not out.strip():break
 except subprocess.TimeoutExpired:pass
 time.sleep(5)
else:raise RuntimeError('GPU cleanup did not finish')
(g/'evidence/service-stop-for-full-fusion.json').write_text(json.dumps({'time':time.time(),'pids':pids,'reason':'completed test; free GPUs for full-fusion contract checks'},indent=2))
command='source /root/gc_moe_isolated_20261116/scripts/00_env.sh; export OMP_NUM_THREADS=2 PYTHONPATH=/root/gc_moe_isolated_20261116/src/vllm-5df63a7:/root/dsk_opt_goal_20261116/src; python -m torch.distributed.run --standalone --nproc-per-node=8 /root/dsk_opt_goal_20261116/scripts/93_humming_full_contract.py'
with (r/'logs/humming-full-contract.log').open('x') as f:p=subprocess.run(['bash','-c',command],stdout=f,stderr=subprocess.STDOUT,timeout=900)
(r/'evidence/humming-full-contract-exit.json').write_text(json.dumps({'exit_code':p.returncode}));print('test exit',p.returncode,flush=True)
