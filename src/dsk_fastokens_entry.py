"""Explicit opt-in entrypoint; records first actual fastokens operations only."""
import os,sys,runpy,json,pathlib,time,importlib.metadata as md
assert os.environ.get('VLLM_USE_FASTOKENS')=='1'
import fastokens
from fastokens._compat import _TokenizerShim
root=pathlib.Path(os.environ['DSK_FASTOKENS_EVIDENCE_DIR']);root.mkdir(parents=True,exist_ok=True)
def record(kind,**kw):
 p=root/f'{os.getpid()}-{kind}.json'
 if not p.exists():p.write_text(json.dumps({'pid':os.getpid(),'kind':kind,'version':md.version('fastokens'),'time':time.time(),**kw},indent=2))
orig_swap=fastokens._swap_backend
def swap(tok,cls):
 result=orig_swap(tok,cls)
 assert isinstance(result._tokenizer,_TokenizerShim)
 record('backend',tokenizer_class=str(type(result)),backend_class=str(type(result._tokenizer)))
 return result
fastokens._swap_backend=swap
for name in ['encode','encode_batch','encode_batch_fast','decode','decode_batch']:
 original=getattr(_TokenizerShim,name)
 def make(n,orig):
  def first(self,*a,**k):
   record(n,backend_class=str(type(self)))
   setattr(_TokenizerShim,n,orig)
   return orig(self,*a,**k)
  return first
 setattr(_TokenizerShim,name,make(name,original))
fastokens.patch_transformers()
if __name__ == '__main__':
 module=sys.argv[1];sys.argv=[module,*sys.argv[2:]]
 runpy.run_module(module,run_name='__main__')
