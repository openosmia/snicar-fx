"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import os
import subprocess
import sys
from pathlib import Path


def test_example_runs(example_script_path):
    """
    Run each example script in a subprocess and verify that the
    script exits with code 0.
    """

    # ensure we have a Path object
    script_path = Path(example_script_path)
    example_dir = script_path.parent

    # set Agg to prevent GUI errors in CI
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(example_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=200,
    )

    # assert
    assert result.returncode == 0, (
        f"Example '{example_dir.name}' failed to run.\n"
        f"Script: {script_path.name}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
