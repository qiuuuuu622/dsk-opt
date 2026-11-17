import pathlib,subprocess,time,json,os,sys
r=pathlib.Path('/root/gc_moe_isolated_20261116'); pid=json.loads((r/'evidence/build-headers-launch.json').read_text())['pid']
proc=pathlib.Path(f'/proc/{pid}')
expected=b'/root/dsk_opt_goal_20261116/scripts/58_e1_build_headers.sh'
if proc.exists():
 assert expected in (proc/'cmdline').read_bytes(), 'Build PID ownership mismatch'
 start=(proc/'stat').read_text().split()[21]
else:
 start=None
while proc.exists():
 try:
  stat=(proc/'stat').read_text().split()
  if stat[21]!=start or stat[2]=='Z':break
 except FileNotFoundError:break
 time.sleep(20)
exitfile=r/'evidence/vllm-build-headers-exit.txt'
assert exitfile.exists() and exitfile.read_text().strip()=='0', 'Build did not complete successfully; no runtime mutation'
os.environ['GC_BUILD_EXIT_FILE']=str(exitfile)
with (r/'logs/runtime-install.log').open('x') as log:
 code=subprocess.call(['bash','/root/dsk_opt_goal_20261116/scripts/53_e1_install_runtime.sh'],stdout=log,stderr=subprocess.STDOUT)
(r/'evidence/runtime-install-exit.json').write_text(json.dumps({'exit_code':code,'build_pid':pid,'build_starttime':start,'controller_pid':os.getpid()},indent=2))
sys.exit(code)
