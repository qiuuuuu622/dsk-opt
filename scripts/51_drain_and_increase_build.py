"""Drain owned Ninja children, then resume incremental build with24 jobs."""
import pathlib,os,signal,subprocess,time,json
r=pathlib.Path('/root/gc_moe_isolated_20261116');pid=359449
p=pathlib.Path(f'/proc/{pid}')
assert (p/'comm').read_text().strip()=='ninja'
assert os.getpgid(pid)==359035
assert str(r) in os.readlink(p/'cwd')
os.kill(pid,signal.SIGSTOP)
children=[int(x) for x in (p/f'task/{pid}/children').read_text().split()]
record={'old_parent':359035,'ninja':pid,'drained_children':children,'reason':'32CPU quota,8compile jobs; increase to24 with idle service and no timed run','started_unix':time.time()}
(r/'evidence/build-job-increase.json').write_text(json.dumps(record,indent=2))
for _ in range(180):
 active=[]
 for child in children:
  file=pathlib.Path(f'/proc/{child}/stat')
  if file.exists() and file.read_text().split(') ',1)[1].split()[0]!='Z':active.append(child)
 if not active:break
 time.sleep(5)
else:
 os.kill(pid,signal.SIGCONT)
 raise TimeoutError('Drain too slow; original build continued')
os.kill(pid,signal.SIGKILL)  # Finished children only; no work-in-progress is killed.
for _ in range(24):
 if (r/'evidence/vllm-build-jobs8-exit.txt').exists():break
 time.sleep(1)
assert (r/'evidence/vllm-build-jobs8-exit.txt').exists()
with (r/'logs/vllm-build-jobs24.log').open('x') as log:
 proc=subprocess.Popen(['bash','/root/dsk_opt_goal_20261116/scripts/50_e1_build_jobs24.sh'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record.update(new_pid=proc.pid,started24_unix=time.time(),old_exit=(r/'evidence/vllm-build-jobs8-exit.txt').read_text().strip(),old_exit_reason='intentional controller interruption after child drain')
(r/'evidence/build-job-increase.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
