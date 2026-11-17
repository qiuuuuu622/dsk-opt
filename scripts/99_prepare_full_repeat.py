import pathlib,shutil,json
r=pathlib.Path('/root/dsk_opt_goal_20261116');old=r/'groups/P-E1-FAST-HUM-FULL-r1';new=r/'groups/P-E1-FAST-HUM-FULL-r2';assert not new.exists()
for d in ['scripts','logs','results','evidence','traces','discovery']:(new/d).mkdir(parents=True)
for p in (old/'scripts').iterdir():
 if p.is_file():(new/'scripts'/p.name).write_text(p.read_text().replace(old.name,new.name).replace('p-e1-fast-hum-full-r1','p-e1-fast-hum-full-r2'))
shutil.copytree(old/'evidence/source-snapshot',new/'evidence/source-snapshot')
(new/'evidence/config-delta.json').write_text(json.dumps({'repeat_of':old.name,'code_and_parameters':'unchanged','reason':'final candidate repeat; fresh worker/cache; original150 ordered0.45QPS'},indent=2))
s=(r/'scripts/96_run_humming_full.py').read_text().replace(old.name,new.name);(r/'scripts/100_run_full_repeat.py').write_text(s)
s=(r/'scripts/97_humming_full_sequence.py').read_text().replace("old='P-E1-FAST-HUM-ALL-r1'","old='P-E1-FAST-HUM-FULL-r1'").replace("[('P-E1-FAST-HUM-FULL-r1','P-E1-FAST-HUM-ALL-r1')]","[('P-E1-FAST-HUM-FULL-r2','P-E1-FAST-HUM-FULL-r1')]").replace('96_run_humming_full.py','100_run_full_repeat.py').replace('evidence/humming-full-sequence.json','evidence/humming-full-repeat-sequence.json');(r/'scripts/101_full_repeat_sequence.py').write_text(s)
