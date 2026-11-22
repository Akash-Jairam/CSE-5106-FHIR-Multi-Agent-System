"""Additive capture of the unchanged final monitor plus stable byte bindings.

Only run after all 2,931 selected outcomes complete. No inference or scorer calls;
the reviewed monitor is invoked without its optional live-journal write flag.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from bound_completion import CAPTURE_METHOD, MONITOR_REL, require, sha, snapshot, validate_proof


def capture(root,run,output):
    root,run,output = map(lambda p:Path(p).resolve(),(root,run,output))
    require(not output.exists() and not output.is_relative_to(run) and not run.is_relative_to(output),
            'Proof capture output must be NEW and outside the run')
    before = snapshot(root,run)
    command = [sys.executable,str(root/MONITOR_REL),'verify','--run-dir',str(run),'--final-arm']
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    process = subprocess.run(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    require(process.returncode==0,'Reviewed final monitor refused completion; no bound proof saved')
    try:
        proof = json.loads(process.stdout)
    except (ValueError,TypeError):
        raise ValueError('Reviewed final monitor returned invalid proof JSON') from None
    after = snapshot(root,run)
    require(before==after,'Completed bytes/inventory changed during final monitor verification')
    validate_proof(proof,before)
    here = Path(__file__).resolve().parent
    binding = {'schema_version':1,'capture_method':CAPTURE_METHOD,
               'captured_at':datetime.now(timezone.utc).isoformat(),'proof_file':'proof.json',
               'proof_sha256':hashlib.sha256(process.stdout).hexdigest(),
               'capture_wrapper_sha256':sha(Path(__file__)),'binding_module_sha256':sha(here/'bound_completion.py'),
               'verifier_command':command,'verifier_exit_code':0,'before':before,'after':after,
               'no_live_journal_append':True,'no_inference_scoring_provider_calls':True}
    output.mkdir(parents=True)
    (output/'proof.json').write_bytes(process.stdout)
    (output/'binding.json').write_text(json.dumps(binding,indent=2,allow_nan=False)+'\n')
    require(sha(output/'proof.json')==binding['proof_sha256'],'Saved raw proof digest differs')
    return {'status':'BOUND_FINAL_MONITOR_PROOF_SAVED','proof':str(output/'proof.json'),
            'binding':str(output/'binding.json'),'selected_outcomes':2931,'source_stable':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',required=True,type=Path)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args = parser.parse_args()
    print(json.dumps(capture(args.project_root,args.run_dir,args.output_dir)))


if __name__=='__main__':
    try:
        main()
    except (ValueError,OSError,KeyError) as error:
        print(json.dumps({'status':'REFUSED','reason':str(error) if isinstance(error,ValueError) else 'Required completed metadata unavailable or malformed'}))
        raise SystemExit(2)
