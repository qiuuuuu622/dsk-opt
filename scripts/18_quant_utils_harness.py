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
sys.path.insert(0,str(pathlib.Path(script).resolve().parent))
if pathlib.Path(script).name == 'test_sm90_gc_moe_se.py':
    namespace = runpy.run_path(script, run_name='gc_test_audited')
    def quant_only_dependency_audit():
        import hashlib
        from gc_moe.runtime.sm90_gc_moe import get_context
        context = get_context()
        return {'gc': namespace['package_audit'](namespace['gc_moe']),
                'deepgemm_C': None, 'deepgemm_C_sha256': None,
                'quant_utils_source': str(p),
                'quant_utils_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                'green_context': {'compute_sms': context.compute_sms,
                                  'dispatch_sms': context.dispatch_sms}}
    namespace['main'].__globals__['dependency_audit'] = quant_only_dependency_audit
    namespace['main']()
else:
    runpy.run_path(script, run_name='__main__')
