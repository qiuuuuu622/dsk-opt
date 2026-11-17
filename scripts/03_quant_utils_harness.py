"""Test-only bridge to archived pure-Python quantizers; no DeepGEMM kernels or ABI loaded."""
import importlib.util,sys,types,runpy,pathlib
root=pathlib.Path('/root/dsk_opt_goal_20261116')
p=root/'evidence/deep_gemm_math.py'
spec=importlib.util.spec_from_file_location('deep_gemm.utils',p)
utils=importlib.util.module_from_spec(spec); spec.loader.exec_module(utils)
pkg=types.ModuleType('deep_gemm'); pkg.__path__=[]; pkg.__file__=str(p)
pkg.__version__='vllm-0.32-vendored-quant-utils-only'; pkg.utils=utils
sys.modules['deep_gemm']=pkg; sys.modules['deep_gemm.utils']=utils
script=sys.argv.pop(1)
if 'readiness' not in script and '--gc-only' not in sys.argv:
 raise RuntimeError('This harness cannot run a DeepGEMM baseline; --gc-only required')
sys.argv[0]=script
runpy.run_path(script,run_name='__main__')
