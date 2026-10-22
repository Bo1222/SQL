"""Local configuration is literal data; no credentials or training are executed."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from train.project_env import load_project_env

ROOT = Path(__file__).resolve().parents[1]


class ProjectEnvTest(unittest.TestCase):
    def test_loads_literal_values_without_expansion_or_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'unexpected-command'
            env_file = Path(directory) / '.env'
            env_file.write_text(
                'WANDB_API_KEY=local-test-key\n'
                'WANDB_MODE=online\n'
                'LITERAL_VARIABLE=${OTHER_VARIABLE}\n'
                f'LITERAL_COMMAND=$(touch {marker})\n'
            )
            with patch.dict(os.environ, {'OTHER_VARIABLE': 'expanded'}, clear=True):
                self.assertTrue(load_project_env(env_file))
                self.assertEqual(os.environ['WANDB_API_KEY'], 'local-test-key')
                self.assertEqual(os.environ['LITERAL_VARIABLE'], '${OTHER_VARIABLE}')
                self.assertEqual(os.environ['LITERAL_COMMAND'], f'$(touch {marker})')
                self.assertFalse(marker.exists())

    def test_existing_environment_has_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / '.env'
            env_file.write_text('WANDB_API_KEY=file-test-key\nWANDB_MODE=online\n')
            with patch.dict(os.environ, {'WANDB_API_KEY': 'process-test-key', 'WANDB_MODE': 'offline'}, clear=True):
                load_project_env(env_file)
                self.assertEqual(os.environ['WANDB_API_KEY'], 'process-test-key')
                self.assertEqual(os.environ['WANDB_MODE'], 'offline')

    def test_missing_file_is_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(load_project_env(Path(directory)/'.env'))

    def test_training_cli_has_priority_and_does_not_store_the_key_in_args(self):
        from train.grpo_ddp import parse_args
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory)/'.env'
            env_file.write_text('WANDB_API_KEY=local-test-key\nWANDB_MODE=online\nWANDB_PROJECT=dotenv-test\n')
            with patch.dict(os.environ, {}, clear=True), \
                 patch('train.grpo_ddp.load_project_env', side_effect=lambda: load_project_env(env_file)), \
                 patch('sys.argv', ['grpo_ddp.py', '--wandb-mode', 'disabled']):
                args=parse_args()
                self.assertEqual(args.wandb_mode,'disabled')
                self.assertEqual(args.wandb_project,'dotenv-test')
                self.assertNotIn('WANDB_API_KEY', vars(args))
                self.assertNotIn('wandb_api_key', vars(args))

    def test_launcher_loads_dotenv_before_external_env_and_only_reexecs_once(self):
        # Copy only setup, never the training/evaluation commands, to a fixture.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'scripts').mkdir()
            (root/'train').mkdir()
            shutil.copy2(ROOT/'train/project_env.py',root/'train/project_env.py')
            (root/'.env').write_text('WANDB_API_KEY=local-test-key\nWANDB_MODE=online\n')
            setup=(ROOT/'scripts/run_grpo_ddp.sh').read_text().split('LOG=',1)[0]
            setup=setup.replace('source /home/ma-user/work/verl_quickstart/scripts/env.sh',
                                'export WANDB_MODE=disabled')
            setup += '\npython3 -c \'import json,os; print(json.dumps({"mode":os.environ["WANDB_MODE"], "key_loaded":os.environ.get("WANDB_API_KEY")=="local-test-key", "loaded_once":os.environ.get("SQLRL_ENV_LOADED")==os.getcwd()}))\'\n'
            script=root/'scripts/run_grpo_ddp.sh'
            script.write_text(setup)
            env=dict(os.environ)
            for key in ['WANDB_MODE','WANDB_API_KEY','SQLRL_ENV_LOADED']:
                env.pop(key,None)
            env.update(NPROC='4',CARDS='0,1,2,3')
            result=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout),{'mode':'online','key_loaded':True,'loaded_once':True})


if __name__ == '__main__':
    unittest.main()
