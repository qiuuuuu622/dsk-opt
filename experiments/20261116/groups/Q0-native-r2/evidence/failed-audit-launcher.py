"""Read-only sampling-parameter audit for diagnostic quality runs only."""
import functools,json,os,runpy
from pathlib import Path
import dynamo.vllm.handlers as handlers
path=Path(os.environ['DSK_SAMPLING_AUDIT'])
path.parent.mkdir(parents=True,exist_ok=True)
handle=path.open('x')
count=0

def wrap(original):
    @functools.wraps(original)
    def audited(*args,**kwargs):
        global count
        result=original(*args,**kwargs)
        count+=1
        values={key:getattr(result,key,None) for key in ['temperature','seed','top_p','top_k','max_tokens','n']}
        handle.write(json.dumps({'call':count,'builder':original.__name__,'effective_sampling_params':values})+'\n');handle.flush()
        return result
    return audited
for name in ['build_sampling_params','build_sampling_params_openai']:
    setattr(handlers,name,wrap(getattr(handlers,name)))
runpy.run_module('dynamo.vllm',run_name='__main__')
