import pathlib,subprocess,json,time,re,os
root=pathlib.Path('/root/dsk_opt_goal_20261116/stability/full-final-candidate');file=root/'thermal.jsonl'
with file.open('x') as f:
 while True:
  state=json.loads((root/'state.json').read_text());pid=state['controller_pid']
  if state['status'] in ['completed','failed'] or not pathlib.Path(f'/proc/{pid}').exists():break
  result={'time':time.time(),'monitor_pid':os.getpid()}
  try:
   raw=subprocess.check_output(['nvidia-smi','-i','2','-q','-d','PERFORMANCE,TEMPERATURE'],text=True,timeout=10)
   result['gpu2_raw']=raw
   for key in ['SW Thermal Slowdown','HW Thermal Slowdown']:
    result[key+'_counter_us']=[int(x) for x in re.findall(re.escape(key)+r'\s*:\s*(\d+) us',raw)]
  except subprocess.TimeoutExpired:result['query_timeout']=True
  f.write(json.dumps(result)+'\n');f.flush();time.sleep(30)
