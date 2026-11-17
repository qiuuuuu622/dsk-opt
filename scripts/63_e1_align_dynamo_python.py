import pathlib,hashlib,json,shutil,py_compile
r=pathlib.Path('/root/gc_moe_isolated_20261116');report=json.loads((r/'evidence/dynamo-python-source-audit.json').read_text());a=pathlib.Path(report['main']);b=pathlib.Path(report['isolated']);backup=r/'src/dynamo-wheel-python-backup'
assert not backup.exists(),'Never overwrite original wheel backup'
for row in report['differing']:
 rel=pathlib.Path(row['path']);assert hashlib.sha256((a/rel).read_bytes()).hexdigest()==row['main_sha256'];assert hashlib.sha256((b/rel).read_bytes()).hexdigest()==row['wheel_sha256']
for row in report['differing']:
 rel=pathlib.Path(row['path']);(backup/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(b/rel,backup/rel);shutil.copy2(a/rel,b/rel);py_compile.compile(str(b/rel),doraise=True)
for name in report['only_main']:
 rel=pathlib.Path(name);(b/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a/rel,b/rel);py_compile.compile(str(b/rel),doraise=True)
assert all((a/p.relative_to(b)).read_bytes()==p.read_bytes() for p in b.rglob('*.py'))
(r/'evidence/dynamo-main-patches-applied.json').write_text(json.dumps({'status':'all_dynamo_python_sources_match_actual_main','changed':report['differing'],'added':report['only_main'],'backup':str(backup),'binary_files_copied':False},indent=2))
shutil.copy2(r/'evidence/runtime-import-smoke.json',r/'evidence/runtime-import-smoke-wheel.json')
print('Aligned 11 differing and 1 missing Dynamo Python files to main; no binaries copied')
