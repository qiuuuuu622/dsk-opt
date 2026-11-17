"""Fail timing admission if any owned E1 build/install process is active."""
import pathlib,os,json
ROOT='/root/gc_moe_isolated_20261116'
def active_build_processes():
 found=[]
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   state=(p/'stat').read_text().split(') ',1)[1].split()[0]
   cwd=str((p/'cwd').resolve())
   comm=(p/'comm').read_text().strip()
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  relevant=(('50_e1_build_jobs24.sh' in argv or '56_e1_build_elfutils.sh' in argv or '58_e1_build_headers.sh' in argv or '59_e1_after_headers.py' in argv or '57_e1_after_elfutils.py' in argv or '55_e1_after_build.py' in argv or '53_e1_install_runtime.sh' in argv) and comm in ('bash','python3','python')) or ((ROOT in argv or cwd.startswith(ROOT)) and comm in ('ninja','cmake','nvcc','cicc','ptxas','cc1plus','gcc','g++','uv')) or ('setup.py build_ext' in argv and cwd.startswith(ROOT))
  if relevant and state not in ('T','t','Z'):found.append({'pid':int(p.name),'state':state,'comm':comm,'argv':argv})
 return found
def assert_timing_clear():
 active=active_build_processes()
 assert not active, 'E1 build/install active; refuse timing: '+json.dumps(active)
if __name__=='__main__': print(json.dumps(active_build_processes(),indent=2))
