"""Plan an isolated runtime from the actual working E0 dependency closure."""
import pathlib,importlib.metadata as md,json,re,subprocess,collections
from packaging.requirements import Requirement
from packaging.markers import default_environment
from packaging.utils import canonicalize_name as canon
r=pathlib.Path('/root/gc_moe_isolated_20261116');src=r/'src/vllm-5df63a7'
main={canon(d.metadata['Name']):md.distribution(d.metadata['Name']) for d in md.distributions()}
iso=json.loads(subprocess.check_output([str(r/'.venv/bin/python'),'-c','import importlib.metadata as m,json;print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()}))'],text=True));iso={canon(k):v for k,v in iso.items()}
roots=[]
for file in ['common.txt','cuda.txt']:
 for line in (src/'requirements'/file).read_text().splitlines():
  line=line.split('#',1)[0].strip()
  if line and not line.startswith('-'):roots.append(Requirement(line))
roots += [Requirement('ai-dynamo==1.5.0'),Requirement('ai-dynamo-runtime==1.5.0')]
env=default_environment();queue=collections.deque(roots);seen=set();selected={};missing=[];violations=[]
while queue:
 req=queue.popleft()
 if req.marker and not req.marker.evaluate({**env,'extra':''}):continue
 name=canon(req.name);key=(name,tuple(sorted(req.extras)))
 if name not in main:
  missing.append(str(req));continue
 d=main[name];selected[name]=d.version
 if req.specifier and not req.specifier.contains(d.version,prereleases=True):violations.append({'requirement':str(req),'main_version':d.version})
 if key in seen:continue
 seen.add(key)
 for raw in d.requires or []:
  dep=Requirement(raw)
  if dep.marker and not any(dep.marker.evaluate({**env,'extra':extra}) for extra in ['']+list(req.extras)):continue
  dep.marker=None
  queue.append(dep)
protected={k:v for k,v in iso.items() if k in selected and k != 'regex'}
selected.update(protected)
selected['torchvision']='0.26.0+cu130'
# Already-installed GC/build packages are untouched. Preserve other actual E0 versions.
install={k:v for k,v in selected.items() if iso.get(k)!=v}
cuda_wheels={k:v for k,v in install.items() if '+cu' in v}
public={k:v for k,v in install.items() if k not in cuda_wheels}
for label,values in [('runtime-frozen-public',public),('runtime-frozen-cuda',cuda_wheels)]:
 (r/'evidence'/f'{label}.txt').write_text(''.join(f'{k}=={v}\n' for k,v in sorted(values.items())))
report={'mode':'freeze actual E0 closure, no-deps install; preserve existing E1 GC/build packages','package_count':len(selected),'install_count':len(install),'public_count':len(public),'cuda_wheels':cuda_wheels,'protected_e1':protected,'missing_from_main':missing,'main_declared_version_mismatches':violations,'planned_differences':{k:{'main':main[k].version if k in main else None,'E1':v} for k,v in selected.items() if k not in main or main[k].version!=v},'pyjwt_pydantic':'Preserve effective E0 PyJWT2.15.1/Pydantic2.14.1. Earlier PyJWT2.7 constraint incorrectly selected lower-priority system dist; fixed metadata precedence. No package mutation occurred.'}
(r/'evidence/runtime-frozen-plan.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
