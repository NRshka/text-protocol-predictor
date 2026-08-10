from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("accelerate") is None, reason="Accelerate CLI unavailable")
def test_two_process_grounding_smoke() -> None:
    result = subprocess.run(
        [
            "accelerate",
            "launch",
            "--multi_gpu",
            "--num_processes",
            "2",
            "--main_process_port",
            "0",
            str(PROJECT_ROOT / "tools" / "accelerate_grounding_smoke.py"),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {"status": "ok", "processes": 2, "finite_losses": 2}
