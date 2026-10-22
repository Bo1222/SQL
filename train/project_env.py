"""Load local project configuration as data, without shell evaluation."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_project_env(path=None):
    """Existing environment variables win; CLI flags are parsed afterwards."""
    env_file = Path(path) if path is not None else PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return False
    from dotenv import load_dotenv

    return load_dotenv(env_file, override=False, interpolate=False)


def main():
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("usage: project_env.py -- COMMAND [ARGS...]")
    load_project_env()
    # Re-exec the launcher once with the loaded environment, before env.sh runs.
    os.environ["SQLRL_ENV_LOADED"] = str(PROJECT_ROOT)
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
