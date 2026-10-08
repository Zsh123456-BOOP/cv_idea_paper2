"""Queue continuation must preserve finished models and record explicit exits."""
import os
import shutil
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def queue(tmp_path):
    scripts=tmp_path/'scripts';scripts.mkdir()
    for name in ('_common.sh','run_restoration_suite.sh'):
        shutil.copy(ROOT/'scripts'/name,scripts/name)
    (tmp_path/'deployment_origin.json').write_text('{}')
    binary=tmp_path/'bin';binary.mkdir()
    nvidia=binary/'nvidia-smi';nvidia.write_text('#!/bin/sh\nexit 1\n');nvidia.chmod(0o755)
    env={**os.environ,'PATH':str(binary)+os.pathsep+os.environ['PATH']}
    out=tmp_path/'lane'
    def run(mode):
        return subprocess.run(['bash',str(scripts/'run_restoration_suite.sh'),'0','data','sources',str(out),mode,'2'],
                              env=env,text=True,capture_output=True)
    return out,run


def test_gpu_query_failure_records_failed_even_without_err_trap(tmp_path):
    out,run=queue(tmp_path)
    result=run('formal')
    assert result.returncode!=0 and (out/'FAILED').is_file()
    assert not (out/'COR-Generator_gold_pixels').exists()


def test_resume_skips_verified_models_without_touching_them(tmp_path):
    out,run=queue(tmp_path);out.mkdir()
    for model in ('COR-Generator','Restormer'):
        dest=out/(model+'_gold_pixels');dest.mkdir()
        (dest/'selection.json').write_text('original selection')
        (dest/'inference_verified.json').write_text('original verification')
        (out/(model+'.log')).write_text('original log')
    result=run('resume')
    assert result.returncode==0,result.stderr
    assert (out/'COMPLETED').is_file()
    assert (out/'COR-Generator_gold_pixels/selection.json').read_text()=='original selection'
    assert (out/'Restormer.log').read_text()=='original log'


def test_resume_refuses_partial_existing_model(tmp_path):
    out,run=queue(tmp_path);out.mkdir()
    dest=out/'COR-Generator_gold_pixels';dest.mkdir();(dest/'partial').write_text('keep')
    result=run('resume')
    assert result.returncode!=0 and (out/'FAILED').is_file()
    assert (dest/'partial').read_text()=='keep'


def test_waits_until_memory_is_available(tmp_path):
    binary=tmp_path/'bin';binary.mkdir();state=tmp_path/'state';state.write_text('busy')
    nvidia=binary/'nvidia-smi'
    nvidia.write_text('#!/bin/sh\nif [ "$(cat "$GPU_TEST_STATE")" = busy ]; then echo 100; else echo 23000; fi\n')
    sleeper=binary/'sleep';sleeper.write_text('#!/bin/sh\necho free > "$GPU_TEST_STATE"\n')
    nvidia.chmod(0o755);sleeper.chmod(0o755)
    env={**os.environ,'PATH':str(binary)+os.pathsep+os.environ['PATH'],'GPU_TEST_STATE':str(state)}
    command='source "$1"; wait_for_gpu_memory fixture 20000 0'
    result=subprocess.run(['bash','-c',command,'test',str(ROOT/'scripts/_common.sh')],env=env,text=True,capture_output=True)
    assert result.returncode==0 and 'waiting for GPU' in result.stdout
    assert state.read_text().strip()=='free'
