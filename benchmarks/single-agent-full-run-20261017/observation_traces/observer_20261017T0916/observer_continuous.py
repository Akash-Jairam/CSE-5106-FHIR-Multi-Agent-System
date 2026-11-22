import argparse,importlib.util,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--until-accepted',type=int,required=True);args=parser.parse_args()
root=Path.cwd();path=root/'benchmarks/single-agent-full-run-20261017/monitor_single_agent.py';spec=importlib.util.spec_from_file_location('single_monitor',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
run=args.run_dir.resolve();snapshot=run/m.AMENDMENT;options=argparse.Namespace(skip_native=False);deadline=time.monotonic()+2400
while time.monotonic()<deadline:
    try:
        r=m.observe(run,snapshot,options);a=r['active_single_agent'] or {};d=a.get('provider_diagnostics') or {}
        row={'at':datetime.now(timezone.utc).isoformat(),'phase':r['state_phase'],'accepted':r['selected_accepted_persisted'],'active_pid':a.get('pid'),'round':a.get('round'),'saved_classifications':a.get('saved_classifications'),'seconds_since_checkpoint':a.get('seconds_since_checkpoint'),'provider_requests':d.get('completed_requests'),'provider_outcomes':d.get('outcomes'),'attention':r['attention'],'disk_free_gib':round(r['disk_free_gib'],2),'unselected_preserved':all(x['unchanged'] for x in r['preserved_unselected'].values()),'native_states':r['current_native'].get('pid_states',{})}
        print(json.dumps(row),flush=True)
        if row['attention'] or r['selected_scope_complete']:
            print(json.dumps({'observer_exit':'attention_or_selected_complete'}),flush=True);break
    except ValueError as e:
        if 'Owned process missing' not in str(e):
            print(json.dumps({'observer_exit':'monitor_refused','reason':str(e)}),flush=True);break
        s=m.read(run/'supervisor_state.json');pid=s['supervisor_pid'];p=subprocess.run(['ps','-p',str(pid),'-o','pid=,stat=,comm='],capture_output=True,text=True);parts=p.stdout.strip().split(None,2)
        if len(parts)!=3 or 'Z' in parts[1]:
            print(json.dumps({'observer_exit':'supervisor_missing_or_zombie','supervisor_pid':pid}),flush=True);break
        print(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'phase':'WAITING_FOR_SUPERVISOR_TRANSITION','supervisor_pid':pid,'supervisor_state':parts[1],'attention':s.get('attention'),'accepted':s.get('accepted',{}).get('single_agent')}),flush=True)
        if s.get('attention'):break
    time.sleep(55)
else:print(json.dumps({'observer_exit':'observation_window_elapsed_not_job_failure'}),flush=True)
