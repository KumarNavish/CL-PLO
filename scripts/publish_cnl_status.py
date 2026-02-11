#!/usr/bin/env python3
"""Collect remote CNL run status over SSH and publish a GitHub-friendly snapshot."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


START_MARKER = "__CNL_STATUS_JSON_START__"
END_MARKER = "__CNL_STATUS_JSON_END__"


def build_remote_probe(model_tag: str, max_log_lines: int) -> str:
    return f"""\
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DATASETS = ["arc_c", "csqa", "medqa", "mmlu"]
USE_FREEZE = [1, 0]
MODEL_TAG = {model_tag!r}
MAX_LOG_LINES = {max_log_lines}


def run_cmd(cmd, timeout=4):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return out.strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {{exc}}"


def tail_lines(path: Path, n: int):
    if not path.exists():
        return []
    try:
        out = subprocess.check_output(
            ["tail", "-n", str(n), str(path)],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:  # noqa: BLE001
        return []
    return [line.rstrip("\\n") for line in out.splitlines()]


def read_last_csv_row(path: Path):
    if not path.exists():
        return {{}}
    last = {{}}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            last = row
    return last


def read_csv_rows(path: Path):
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            out.append(row)
    return out


def detect_status(log_path: Path):
    if not log_path.exists():
        return "pending"
    text = "\\n".join(tail_lines(log_path, 120))
    if "Traceback (most recent call last):" in text or "CUDA out of memory" in text or "Error:" in text:
        return "error"
    if "Done." in text or "Completed full SmolLM CNL+vanilla batch." in text:
        return "done"
    return "running"


def latest_epoch(log_path: Path):
    if not log_path.exists():
        return ""
    epoch = ""
    pattern = re.compile(r"^===== Epoch ([0-9]+) =====$")
    for line in tail_lines(log_path, 400):
        m = pattern.match(line)
        if m:
            epoch = m.group(1)
    return epoch


def read_elapsed(path: Path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip().replace("ELAPSED:", "")


def parse_gpu_rows():
    out = run_cmd([
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader",
    ])
    rows = []
    if out.startswith("ERROR:"):
        return rows
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {{
                "gpu": parts[0],
                "name": parts[1],
                "memory_used": parts[2],
                "memory_total": parts[3],
                "utilization": parts[4],
            }}
        )
    return rows


def latest_run_log():
    logs = sorted(Path("logs").glob("full_smollm_baselines_*.out"), reverse=True)
    return logs[0] if logs else None


def collect_trycloudflare_url():
    regex = re.compile(r"https://[-a-z0-9]+\\.trycloudflare\\.com")
    for path in sorted(Path("logs").glob("cloudflared_dashboard_*.out"), reverse=True):
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = regex.findall(text)
        if matches:
            return matches[-1], str(path)
    return "", ""


jobs = []
for ds in DATASETS:
    for uf in USE_FREEZE:
        log_path = Path(f"logs/train_{{ds}}_usefreeze{{uf}}.log")
        time_path = Path(f"logs/train_{{ds}}_usefreeze{{uf}}.time")
        summary_path = Path(f"zero_ckpts/{{ds}}_{{MODEL_TAG}}_lr1e-7_usefreeze{{uf}}/summary.csv")
        last = read_last_csv_row(summary_path)
        jobs.append(
            {{
                "dataset": ds,
                "use_freeze": uf,
                "status": detect_status(log_path),
                "epoch": last.get("epoch") or latest_epoch(log_path),
                "train_avg_loss": last.get("train_avg_loss", ""),
                "wrong_to_correct": last.get("wrong_to_correct", ""),
                "correct_to_wrong": last.get("correct_to_wrong", ""),
                "elapsed": read_elapsed(time_path),
                "summary_path": str(summary_path),
                "log_path": str(log_path),
            }}
        )

run_log = latest_run_log()
run_log_path = str(run_log) if run_log else ""
run_log_tail = tail_lines(run_log, MAX_LOG_LINES) if run_log else []
cloudflare_url, cloudflare_log = collect_trycloudflare_url()
merged_path = Path("results/smollm_full_baselines_summary.csv")

out = {{
    "collected_at_utc": datetime.now(timezone.utc).isoformat(),
    "host": run_cmd(["hostname"]),
    "cwd": str(Path(".").resolve()),
    "model_tag": MODEL_TAG,
    "gpu": parse_gpu_rows(),
    "jobs": jobs,
    "coordinator_log_path": run_log_path,
    "coordinator_log_tail": run_log_tail,
    "merged_summary_path": str(merged_path),
    "merged_summary_rows": read_csv_rows(merged_path),
    "processes": {{
        "runner": run_cmd(["pgrep", "-af", "run_full_smollm_baselines_remote.sh"]),
        "train": run_cmd(["pgrep", "-af", "python3 sft/sft.py"]),
        "infer": run_cmd(["pgrep", "-af", "python3 infer/infer.py"]),
        "dashboard": run_cmd(["pgrep", "-af", "python3 scripts/live_dashboard.py"]),
        "cloudflared": run_cmd(["pgrep", "-af", "cloudflared tunnel"]),
    }},
    "public_dashboard_url": cloudflare_url,
    "public_dashboard_source_log": cloudflare_log,
    "public_dashboard_http_noauth": run_cmd(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{{http_code}}", cloudflare_url + "/"]
    )
    if cloudflare_url
    else "",
}}

print({START_MARKER!r})
print(json.dumps(out, indent=2, ensure_ascii=True))
print({END_MARKER!r})
"""


def extract_marked_json(stdout: str) -> dict[str, Any]:
    start_idx = stdout.find(START_MARKER)
    end_idx = stdout.find(END_MARKER)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise RuntimeError("Probe output missing JSON markers.")
    payload = stdout[start_idx + len(START_MARKER) : end_idx].strip()
    return json.loads(payload)


def sanitize_text(text: str) -> str:
    patterns = [
        (re.compile(r"(--auth_password\s+)(\S+)"), r"\1***"),
        (re.compile(r"(DASHBOARD_PASSWORD=)(\S+)"), r"\1***"),
        (re.compile(r"(-u\s+[^:\s]+:)(\S+)"), r"\1***"),
    ]
    out = text
    for pattern, repl in patterns:
        out = pattern.sub(repl, out)
    return out


def sanitize_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_obj(v) for v in value]
    if isinstance(value, str):
        return sanitize_text(value)
    return value


def write_html(path: Path, json_path: str) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CNL SmolLM Remote Dashboard</title>
  <style>
    :root {{
      --bg: #f4f8fb;
      --ink: #11202d;
      --muted: #4e6475;
      --card: #ffffff;
      --line: #d6e2eb;
      --good: #17622b;
      --warn: #925b00;
      --bad: #9f1d2d;
    }}
    body {{
      margin: 0;
      background: linear-gradient(165deg, #eef5fa 0%, #f9fcff 70%);
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
    }}
    .wrap {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0.4px;
    }}
    .muted {{
      color: var(--muted);
      margin: 0 0 14px;
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      margin-bottom: 14px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 8px 20px rgba(26, 57, 82, 0.07);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eaf2f8;
      font-size: 12px;
      letter-spacing: 0.3px;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    .status-running {{ color: var(--warn); font-weight: 700; }}
    .status-done {{ color: var(--good); font-weight: 700; }}
    .status-error {{ color: var(--bad); font-weight: 700; }}
    .status-pending {{ color: var(--muted); font-weight: 700; }}
    pre {{
      margin: 0;
      background: #0f172a;
      color: #d9e5f2;
      border-radius: 10px;
      padding: 12px;
      overflow: auto;
      max-height: 330px;
      font-size: 12px;
      line-height: 1.35;
    }}
    .row {{
      display: grid;
      gap: 14px;
      grid-template-columns: 1fr 1fr;
      margin-top: 14px;
    }}
    @media (max-width: 900px) {{
      .row {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>CNL SmolLM Remote Dashboard</h1>
    <p class="muted" id="stamp">Loading snapshot...</p>
    <div class="grid">
      <div class="card">
        <strong>Remote Host</strong>
        <div id="host">-</div>
      </div>
      <div class="card">
        <strong>Model Tag</strong>
        <div id="model">-</div>
      </div>
      <div class="card">
        <strong>Public Live Dashboard</strong>
        <div id="public_url">-</div>
      </div>
    </div>

    <h2>Jobs</h2>
    <table id="jobs_table"></table>

    <h2 style="margin-top: 14px;">GPU</h2>
    <table id="gpu_table"></table>

    <div class="row">
      <div class="card">
        <strong>Coordinator Log Tail</strong>
        <p class="muted" id="log_path">-</p>
        <pre id="run_log"></pre>
      </div>
      <div class="card">
        <strong>Processes</strong>
        <pre id="procs"></pre>
      </div>
    </div>
  </div>

  <script>
    const JSON_PATH = {json_path!r};
    const STATUS_ORDER = ["running", "done", "error", "pending"];

    function esc(x) {{
      return String(x ?? "").replace(/[&<>]/g, (c) => {{
        if (c === "&") return "&amp;";
        if (c === "<") return "&lt;";
        return "&gt;";
      }});
    }}

    function toRows(headers, rows) {{
      let html = "<tr>" + headers.map((h) => `<th>${{esc(h.label)}}</th>`).join("") + "</tr>";
      for (const row of rows) {{
        html += "<tr>" + headers.map((h) => {{
          const value = row[h.key] ?? "";
          if (h.key === "status") {{
            const cls = `status-${{value}}`;
            return `<td class="${{esc(cls)}}">${{esc(value)}}</td>`;
          }}
          return `<td>${{esc(value)}}</td>`;
        }}).join("") + "</tr>";
      }}
      return html;
    }}

    async function load() {{
      const res = await fetch(`${{JSON_PATH}}?t=${{Date.now()}}`, {{ cache: "no-store" }});
      const data = await res.json();
      document.getElementById("stamp").textContent =
        `Snapshot updated: ${{data.collected_at_utc || "-"}}`;
      document.getElementById("host").textContent = data.host || "-";
      document.getElementById("model").textContent = data.model_tag || "-";
      const url = data.public_dashboard_url || "";
      document.getElementById("public_url").innerHTML = url
        ? `<a href="${{esc(url)}}" target="_blank" rel="noopener noreferrer">${{esc(url)}}</a>`
        : "not available";

      const jobs = [...(data.jobs || [])].sort((a, b) => {{
        const sa = STATUS_ORDER.indexOf(a.status);
        const sb = STATUS_ORDER.indexOf(b.status);
        if (sa !== sb) return sa - sb;
        return `${{a.dataset}}:${{a.use_freeze}}`.localeCompare(`${{b.dataset}}:${{b.use_freeze}}`);
      }});
      document.getElementById("jobs_table").innerHTML = toRows(
        [
          {{ key: "dataset", label: "dataset" }},
          {{ key: "use_freeze", label: "use_freeze" }},
          {{ key: "status", label: "status" }},
          {{ key: "epoch", label: "epoch" }},
          {{ key: "train_avg_loss", label: "train_avg_loss" }},
          {{ key: "wrong_to_correct", label: "wrong_to_correct" }},
          {{ key: "correct_to_wrong", label: "correct_to_wrong" }},
          {{ key: "elapsed", label: "elapsed" }}
        ],
        jobs
      );

      document.getElementById("gpu_table").innerHTML = toRows(
        [
          {{ key: "gpu", label: "gpu" }},
          {{ key: "name", label: "name" }},
          {{ key: "memory_used", label: "memory_used" }},
          {{ key: "memory_total", label: "memory_total" }},
          {{ key: "utilization", label: "utilization" }}
        ],
        data.gpu || []
      );

      document.getElementById("log_path").textContent = data.coordinator_log_path || "-";
      document.getElementById("run_log").textContent = (data.coordinator_log_tail || []).join("\\n");
      const procs = data.processes || {{}};
      document.getElementById("procs").textContent =
        `runner:\\n${{procs.runner || ""}}\\n\\ntrain:\\n${{procs.train || ""}}\\n\\ninfer:\\n${{procs.infer || ""}}\\n\\ndashboard:\\n${{procs.dashboard || ""}}\\n\\ncloudflared:\\n${{procs.cloudflared || ""}}`;
    }}

    async function tick() {{
      try {{
        await load();
      }} catch (err) {{
        document.getElementById("stamp").textContent = `Load error: ${{err}}`;
      }}
    }}

    tick();
    setInterval(tick, 20000);
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def collect_snapshot(remote_host: str, remote_dir: str, model_tag: str, max_log_lines: int) -> dict[str, Any]:
    probe = build_remote_probe(model_tag=model_tag, max_log_lines=max_log_lines)
    with tempfile.TemporaryDirectory(prefix="cnl_status_") as tmpdir:
        probe_path = Path(tmpdir) / "probe.py"
        out_path = Path(tmpdir) / "stdout.txt"
        err_path = Path(tmpdir) / "stderr.txt"
        probe_path.write_text(probe, encoding="utf-8")

        shell_cmd = (
            f"cat {shlex.quote(str(probe_path))} | "
            f"ssh -o BatchMode=yes {shlex.quote(remote_host)} "
            f"\"cd {shlex.quote(remote_dir)} && python3 -\" "
            f"> {shlex.quote(str(out_path))} 2> {shlex.quote(str(err_path))}"
        )
        proc = subprocess.run(
            ["/bin/zsh", "-lc", shell_cmd],
            text=True,
            capture_output=False,
            check=False,
        )
        stdout = out_path.read_text(encoding="utf-8", errors="replace")
        stderr = err_path.read_text(encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Remote probe failed (code {proc.returncode}). stderr:\\n{stderr.strip()}"
            )
        return extract_marked_json(stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-host", default="fibonacci")
    parser.add_argument("--remote-dir", default="~/Collaborative-Neuron-Learning")
    parser.add_argument("--model-tag", default="SmolLM2-360M-Instruct")
    parser.add_argument("--max-log-lines", type=int, default=80)
    parser.add_argument("--output-json", default="data/cnl_status.json")
    parser.add_argument("--output-html", default="cnl-dashboard.html")
    parser.add_argument("--html-json-path", default="data/cnl_status.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_json = Path(args.output_json)
    out_html = Path(args.output_html)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    data = collect_snapshot(
        remote_host=args.remote_host,
        remote_dir=args.remote_dir,
        model_tag=args.model_tag,
        max_log_lines=args.max_log_lines,
    )
    data = sanitize_obj(data)
    data["publisher_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    out_json.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_html(out_html, args.html_json_path)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_html}")
    print(f"Remote host: {data.get('host', '-')}")
    print(f"Collected at: {data.get('collected_at_utc', '-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
