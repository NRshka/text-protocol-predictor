from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_example_generator_passes_production_validation(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "create_mask_grounding_example.py"),
            str(tmp_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "validate_mask_dataset.py"),
            "--dataset-root",
            str(tmp_path),
            "--manifest",
            "manifest.jsonl",
        ],
        check=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report == {
        "status": "ok",
        "samples": 1,
        "text_objects": 2,
        "supervised_objects": 2,
        "samples_by_mask_source": {"files": 1},
    }
