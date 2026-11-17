import pathlib,json,os,signal,time,subprocess
r=pathlib.Path('/root/dsk_opt_goal_20261116');g=r/'groups/P-GC-ALL-w4a8-r3'
for pid,needle in [(447575,'76_run_gc_aligned_frontend'),(447653,'77_finalize_gc')]:
 p=pathlib.Path(f'/proc/{pid}/cmdline')
 if p.exists():
  assert needle in p.read_text();os.kill(pid,signal.SIGTERM)
pids=json.loads((g/'evidence/service-pids.json').read_text())
for pid in pids.values():
 try:assert os.getpgid(pid)==pid;os.killpg(pid,signal.SIGTERM)
 except ProcessLookupError:pass
time.sleep(20)
for pid in pids.values():
 try:os.killpg(pid,signal.SIGKILL)
 except ProcessLookupError:pass
(g/'evidence/run-state.json').write_text(json.dumps({'status':'failed','reason':'GC native child CUDA graph instantiate OOM during formal replay; 53 completed at detection; no valid full-run comparison','pids':pids},indent=2))
p=r/'src/dsk_gc_adapter.py';s=p.read_text();(g/'evidence/dsk_gc_adapter-before-graph-fix.py').write_text(s)
s=s.replace('_BUFFERS = {}','_BUFFERS = {}\n_OUTPUTS = {}')
s=s.replace('self.buffer=_BUFFERS[key]','self.buffer=_BUFFERS[key]\n        if key not in _OUTPUTS:\n            _OUTPUTS[key]=torch.empty((capacity,hidden),device="cuda",dtype=torch.bfloat16)\n        self.output=_OUTPUTS[key]')
s=s.replace('chunk=triton.cdiv(m,self.ranks);b=self.buffer','chunk=triton.next_power_of_2(triton.cdiv(m,self.ranks));b=self.buffer\n        assert chunk<=self.output.shape[0]')
s=s.replace('y=torch.empty((chunk,h),device=x.device,dtype=torch.bfloat16)','y=self.output[:chunk]')
p.write_text(s)
s=(r/'scripts/70_gc_adapter_triton37_verify.py').read_text().replace('results/gc-service-adapter-triton37','results/gc-service-adapter-stable-graph')
s=s.replace("assert a.buffer is b.buffer","assert a.buffer is b.buffer and a.output is b.output")
(r/'scripts/79_gc_stable_graph_verify.py').write_text(s)
for _ in range(24):
 try:
  out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10)
  if not out.strip():break
 except subprocess.TimeoutExpired:pass
 time.sleep(5)
else:raise RuntimeError('GPU cleanup not completed')
command='source /root/gc_moe_isolated_20261116/scripts/00_env.sh; export OMP_NUM_THREADS=2 PYTHONPATH=/root/gc_moe_isolated_20261116/src/vllm-5df63a7:/root/dsk_opt_goal_20261116/src; python -m torch.distributed.run --standalone --nproc-per-node=8 /root/dsk_opt_goal_20261116/scripts/79_gc_stable_graph_verify.py'
with (r/'logs/gc-stable-graph-verify.log').open('x') as f:code=subprocess.call(['bash','-c',command],stdout=f,stderr=subprocess.STDOUT)
(r/'evidence/gc-stable-graph-exit.txt').write_text(str(code));print('test exit',code,flush=True)
