"""Check launch parsing and environment setup without running the train script."""
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from torch.distributed.run import get_args_parser
from train.grpo_ddp import parse_args

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / 'scripts/run_grpo_ddp.sh').read_text()


class GrpoLauncherTest(unittest.TestCase):
    def test_actual_launch_command_parses_through_both_parsers(self):
        logical = SCRIPT.replace('\\\n', ' ')
        line = next(line for line in logical.splitlines() if line.startswith('torchrun '))
        tokens = shlex.split(line)
        tokens = tokens[1:tokens.index('2>&1')]
        replacements = {'$NPROC': '4', '${MASTER_PORT:-29511}': '29511',
                        '$WANDB_MODE': 'offline', '$WANDB_PROJECT': 'sqlrl', '$WANDB_NAME': 'test'}
        for old, new in replacements.items():
            tokens = [token.replace(old, new) for token in tokens]
        launched = get_args_parser().parse_args(tokens)
        self.assertEqual(launched.training_script, 'train/grpo_ddp.py')
        self.assertEqual(launched.master_port, 29511)
        with patch('sys.argv', ['grpo_ddp.py'] + launched.training_script_args):
            args = parse_args()
        self.assertEqual(args.n, 8)
        self.assertEqual(args.wandb_mode, 'offline')

    def run_setup_only(self, mode=None, nproc='4'):
        # Stop before LOG and every launcher/evaluation command. Model the exact
        # hostile setting in the external env.sh without sourcing that environment.
        setup = SCRIPT.split('LOG=', 1)[0]
        setup = setup.replace('source /home/ma-user/work/verl_quickstart/scripts/env.sh',
                              'export WANDB_MODE=disabled')
        setup = '\n'.join('GRPO_ROOT="$GRPO_TEST_ROOT"' if line.startswith('GRPO_ROOT=') else line for line in setup.splitlines())
        setup += '\nprintf "%s" "$WANDB_MODE"\n'
        env = dict(os.environ)
        env.pop('WANDB_MODE', None)
        env.update(NPROC=nproc, CARDS='0,1,2,3')
        if mode is not None:
            env['WANDB_MODE'] = mode
        with tempfile.TemporaryDirectory() as directory:
            env['GRPO_TEST_ROOT'] = directory
            env['SQLRL_ENV_LOADED'] = directory
            return subprocess.run(['bash', '-c', setup], env=env, text=True,
                                  capture_output=True, check=False)

    def test_external_disabled_setting_does_not_override_requested_mode(self):
        for requested, expected in [(None, 'offline'), ('online', 'online'), ('disabled', 'disabled')]:
            with self.subTest(mode=requested):
                result = self.run_setup_only(requested)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)

    def test_invalid_worker_counts_rejected_before_launch(self):
        for nproc in ['0', 'invalid', '5']:
            with self.subTest(nproc=nproc):
                result = self.run_setup_only(nproc=nproc)
                self.assertEqual(result.returncode, 2)
                self.assertTrue(result.stderr)


if __name__ == '__main__':
    unittest.main()
