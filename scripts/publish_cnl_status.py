#!/usr/bin/env python3
"""Backward-compatible wrapper for legacy CNL snapshot publishing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-host", default="fibonacci")
    parser.add_argument(
        "--remote-dir",
        default="/users/staff/dmi-dmi/kumar0002/Collaborative-Neuron-Learning",
    )
    parser.add_argument("--model-tag", default="SmolLM2-360M-Instruct")
    parser.add_argument("--max-log-lines", type=int, default=80)
    parser.add_argument("--output-json", default="data/cnl_status.json")
    parser.add_argument("--output-html", default="cnl-dashboard.html")
    parser.add_argument("--html-json-path", default="data/run_status.json")
    parser.add_argument("--config", default="configs/remote_status_projects.json")
    parser.add_argument("--project-id", default="fibonacci-cnl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_config_path = Path(args.config)
    config = json.loads(base_config_path.read_text(encoding="utf-8"))

    project = None
    for item in config.get("projects", []):
        if item.get("id") == args.project_id:
            project = item
            break
    if project is None:
        raise RuntimeError(f"Project id not found in config: {args.project_id}")

    project.setdefault("remote", {})
    project["remote"]["host"] = args.remote_host
    project["remote"]["workdir"] = args.remote_dir
    project.setdefault("settings", {})
    project["settings"]["model_tag"] = args.model_tag

    with tempfile.TemporaryDirectory(prefix="cnl_wrapper_") as tmpdir:
        tmp_cfg = Path(tmpdir) / "config.json"
        tmp_cfg.write_text(json.dumps(config, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

        cmd = [
            sys.executable,
            "scripts/publish_remote_status.py",
            "--config",
            str(tmp_cfg),
            "--project-ids",
            args.project_id,
            "--output-json",
            "data/run_status.json",
            "--output-html",
            args.output_html,
            "--html-json-path",
            args.html_json_path,
            "--max-log-lines",
            str(args.max_log_lines),
            "--compat-cnl-json",
            args.output_json,
            "--dashboard-title",
            "CNL SmolLM Remote Dashboard",
        ]

        proc = subprocess.run(cmd, check=False)
        return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
