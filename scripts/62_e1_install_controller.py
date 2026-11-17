import subprocess,pathlib,json,os
r=pathlib.Path('/root/gc_moe_isolated_20261116')
env={**os.environ,'GC_BUILD_EXIT_FILE':str(r/'evidence/vllm-build-headers-exit.txt')}
with (r/'logs/runtime-install-warm-cache.log').open('x') as log:
 code=subprocess.call(['bash','/root/dsk_opt_goal_20261116/scripts/61_e1_install_warm_cache.sh'],stdout=log,stderr=subprocess.STDOUT,env=env)
(r/'evidence/runtime-install-warm-cache-exit.json').write_text(json.dumps({'exit_code':code,'controller_pid':os.getpid()},indent=2))
raise SystemExit(code)
