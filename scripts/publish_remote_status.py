#!/usr/bin/env python3
"""Config-driven remote status publisher for SSH-accessible projects."""

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


START_MARKER = "__REMOTE_STATUS_JSON_START__"
END_MARKER = "__REMOTE_STATUS_JSON_END__"


REMOTE_PROBE_TEMPLATE = r"""#!/usr/bin/env python3
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PAYLOAD = json.loads(__PAYLOAD_JSON__)
PROJECT = PAYLOAD["project"]
GLOBAL_MAX_LOG_LINES = int(PAYLOAD.get("max_log_lines", 80))


def run_cmd(cmd, timeout=5, shell=False):
    try:
        if shell:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout, shell=True)
        else:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return out.strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def tail_lines(path_obj, n):
    path_obj = Path(path_obj)
    if not path_obj.exists():
        return []
    try:
        out = subprocess.check_output(
            ["tail", "-n", str(n), str(path_obj)],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:  # noqa: BLE001
        return []
    return [line.rstrip("\n") for line in out.splitlines()]


def read_csv_rows(path_obj, limit=1000):
    path_obj = Path(path_obj)
    if not path_obj.exists():
        return []
    rows = []
    with path_obj.open("r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if limit > 0:
        return rows[-limit:]
    return rows


def read_last_csv_row(path_obj):
    rows = read_csv_rows(path_obj, limit=1)
    return rows[0] if rows else {}


def read_elapsed(path_obj):
    path_obj = Path(path_obj)
    if not path_obj.exists():
        return ""
    txt = path_obj.read_text(encoding="utf-8", errors="replace").strip()
    return txt.replace("ELAPSED:", "")


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def fmt(template, values):
    if not template:
        return ""
    return template.format_map(SafeDict(values))


def expand_dimensions(dimensions):
    dims = dimensions or {}
    keys = list(dims.keys())
    if not keys:
        return [{}]
    combos = [{}]
    for key in keys:
        vals = dims.get(key, [])
        if not isinstance(vals, list):
            vals = [vals]
        nxt = []
        for combo in combos:
            for val in vals:
                row = dict(combo)
                row[key] = val
                nxt.append(row)
        combos = nxt
    return combos


def detect_status(log_path, error_markers, done_markers):
    log_path = Path(log_path)
    if not log_path.exists():
        return "pending"
    text = "\n".join(tail_lines(log_path, 120))
    for marker in error_markers or []:
        if marker and marker in text:
            return "error"
    for marker in done_markers or []:
        if marker and marker in text:
            return "done"
    return "running"


def latest_epoch(log_path, epoch_regex):
    log_path = Path(log_path)
    if not log_path.exists():
        return ""
    pattern = re.compile(epoch_regex) if epoch_regex else None
    epoch = ""
    for line in tail_lines(log_path, 500):
        if pattern:
            m = pattern.search(line)
            if m:
                epoch = m.group(1)
    return epoch


def collect_gpu_table():
    out = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ]
    )
    if out.startswith("ERROR:"):
        return {
            "title": "GPU",
            "columns": [{"key": "error", "label": "error"}],
            "rows": [{"error": out}],
        }
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "gpu": parts[0],
                "name": parts[1],
                "memory_used": parts[2],
                "memory_total": parts[3],
                "utilization": parts[4],
            }
        )
    return {
        "title": "GPU",
        "columns": [
            {"key": "gpu", "label": "gpu"},
            {"key": "name", "label": "name"},
            {"key": "memory_used", "label": "memory_used"},
            {"key": "memory_total", "label": "memory_total"},
            {"key": "utilization", "label": "utilization"},
        ],
        "rows": rows,
    }


def collect_csv_table(title, path_value, mode="all", limit=100):
    path_obj = Path(path_value)
    if mode == "last_row":
        rows = []
        last = read_last_csv_row(path_obj)
        if last:
            rows = [last]
    else:
        rows = read_csv_rows(path_obj, limit=limit)
    columns = []
    if rows:
        columns = [{"key": k, "label": k} for k in rows[0].keys()]
    return {"title": title, "columns": columns, "rows": rows}


def normalize_columns(columns, rows):
    if columns:
        out = []
        for col in columns:
            if isinstance(col, str):
                out.append({"key": col, "label": col})
            elif isinstance(col, dict):
                key = col.get("key", "")
                out.append({"key": key, "label": col.get("label", key)})
        return out
    if rows:
        return [{"key": k, "label": k} for k in rows[0].keys()]
    return []


def collect_matrix_table(matrix_cfg, shared_values):
    title = matrix_cfg.get("title", "Jobs")
    dims = matrix_cfg.get("dimensions", {})
    combinations = expand_dimensions(dims)
    error_markers = matrix_cfg.get("error_markers", ["Traceback (most recent call last):", "CUDA out of memory", "Error:"])
    done_markers = matrix_cfg.get("done_markers", ["Done."])
    epoch_regex = matrix_cfg.get("epoch_regex", r"^===== Epoch ([0-9]+) =====$")
    metrics = matrix_cfg.get("metrics", [])
    rows = []

    for combo in combinations:
        values = dict(shared_values)
        values.update(combo)
        log_path = fmt(matrix_cfg.get("log_path_template", ""), values)
        summary_csv = fmt(matrix_cfg.get("summary_csv_template", ""), values)
        time_path = fmt(matrix_cfg.get("time_path_template", ""), values)
        summary_row = read_last_csv_row(summary_csv) if summary_csv else {}

        row = dict(combo)
        row["status"] = detect_status(log_path, error_markers, done_markers)
        row["epoch"] = summary_row.get("epoch") or latest_epoch(log_path, epoch_regex)

        if metrics:
            for key in metrics:
                row[key] = summary_row.get(key, "")
        elif summary_row:
            for key in summary_row:
                if key not in row and key != "epoch":
                    row[key] = summary_row.get(key, "")

        if time_path:
            row["elapsed"] = read_elapsed(time_path)

        row["log_path"] = log_path
        row["summary_path"] = summary_csv
        rows.append(row)

    columns = normalize_columns(matrix_cfg.get("columns"), rows)
    if not columns:
        keys = list((rows[0] if rows else {}).keys())
        hidden = {"log_path", "summary_path"}
        columns = [{"key": k, "label": k} for k in keys if k not in hidden]
    return {"title": title, "columns": columns, "rows": rows}


def to_int(value, default=0):
    try:
        return int(str(value).strip())
    except Exception:  # noqa: BLE001
        return default


def to_float(value, default=0.0):
    try:
        return float(str(value).strip())
    except Exception:  # noqa: BLE001
        return default


def count_nonempty_lines(path_obj):
    path_obj = Path(path_obj)
    if not path_obj.exists():
        return 0
    count = 0
    with path_obj.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def dataset_title(dataset):
    names = {
        "arc_c": "Arc-c",
        "csqa": "CSQA",
        "mmlu": "MMLU",
        "medqa": "MEDQA",
    }
    return names.get(dataset, dataset)


def pct_text(numerator, denominator):
    if denominator <= 0:
        return "0.00%"
    return f"{(100.0 * numerator / denominator):.2f}%"


def read_jsonl_rows(path_obj):
    path_obj = Path(path_obj)
    if not path_obj.exists():
        return []
    rows = []
    with path_obj.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return rows


def build_cnl_table1(settings, jobs_rows, model_tag):
    datasets = settings.get("datasets", ["arc_c", "csqa", "medqa", "mmlu"])
    ft_use_freeze = int(settings.get("table1_ft_use_freeze", 0))
    target_epoch = int(settings.get("table1_target_epoch", 25))

    by_pair = {}
    for row in jobs_rows:
        ds = str(row.get("dataset", ""))
        uf = to_int(row.get("use_freeze"), default=None)
        if ds and uf is not None:
            by_pair[(ds, uf)] = row

    ready_datasets = []
    pending_datasets = []
    neg_totals = {}
    values = {}

    for ds in datasets:
        ft_row = by_pair.get((ds, ft_use_freeze))
        ft_epoch = to_int((ft_row or {}).get("epoch"), default=0)
        if ft_epoch < target_epoch:
            pending_datasets.append(ds)
            continue

        distance_jsonl = Path(f"distance/{ds}_{model_tag}/correct_with_grad_dot.jsonl")
        ft_correct_jsonl = Path(
            f"zero_ckpts/{ds}_{model_tag}_lr1e-7_usefreeze{ft_use_freeze}/jsonl/infer_correct_ep{target_epoch}.jsonl"
        )
        if not distance_jsonl.exists() or not ft_correct_jsonl.exists():
            pending_datasets.append(ds)
            continue

        dist_rows = read_jsonl_rows(distance_jsonl)
        neg_rows = [r for r in dist_rows if to_float(r.get("grad_dot"), default=0.0) < 0.0]
        neg_rows.sort(key=lambda r: abs(to_float(r.get("grad_dot"), default=0.0)), reverse=True)
        n_neg = len(neg_rows)
        third = n_neg // 3
        sim_rows = neg_rows[:third]
        dissim_rows = neg_rows[-third:] if third > 0 else []

        ft_rows = read_jsonl_rows(ft_correct_jsonl)
        forgot_map = {}
        for r in ft_rows:
            q = str(r.get("question", ""))
            forgot_map[q] = str(r.get("label", "")) != str(r.get("predict_label", ""))

        sim_forgot = sum(1 for r in sim_rows if forgot_map.get(str(r.get("question", "")), False))
        dissim_forgot = sum(1 for r in dissim_rows if forgot_map.get(str(r.get("question", "")), False))

        values[ds] = {
            "dissimilar": f"{dissim_forgot} ({pct_text(dissim_forgot, n_neg)})",
            "similar": f"{sim_forgot} ({pct_text(sim_forgot, n_neg)})",
        }
        neg_totals[ds] = n_neg
        ready_datasets.append(ds)

    if not ready_datasets:
        return None, ready_datasets, pending_datasets, neg_totals

    columns = [{"key": "model", "label": "MODEL"}]
    for ds in ready_datasets:
        columns.append({"key": f"{ds}_dissimilar", "label": f"{dataset_title(ds)} Dissimilar"})
        columns.append({"key": f"{ds}_similar", "label": f"{dataset_title(ds)} Similar"})

    row = {"model": "SmolLM-360M"}
    for ds in ready_datasets:
        row[f"{ds}_dissimilar"] = values[ds]["dissimilar"]
        row[f"{ds}_similar"] = values[ds]["similar"]

    table = {
        "title": "Table 1 (SmolLM): Similar vs Dissimilar",
        "columns": columns,
        "rows": [row],
    }
    return table, ready_datasets, pending_datasets, neg_totals


def build_cnl_table2(settings, model_tag):
    datasets = settings.get("datasets", ["arc_c", "csqa", "medqa", "mmlu"])
    ready_datasets = []
    pending_datasets = []
    values = {}

    for ds in datasets:
        csv_path = Path(f"neuron/{ds}_{model_tag}/col_conf_distri.csv")
        if not csv_path.exists():
            pending_datasets.append(ds)
            continue
        rows = read_csv_rows(csv_path, limit=1)
        if not rows:
            pending_datasets.append(ds)
            continue
        values[ds] = rows[0]
        ready_datasets.append(ds)

    if not ready_datasets:
        return None, ready_datasets, pending_datasets

    columns = [
        {"key": "model", "label": "MODEL"},
        {"key": "neuron_type", "label": "Neuron Type"},
    ]
    for ds in ready_datasets:
        columns.append({"key": f"{ds}_stats", "label": f"{dataset_title(ds)} (Prop/Grad/Total)"})

    coll_row = {"model": "SmolLM-360M", "neuron_type": "Collaborative"}
    conf_row = {"model": "", "neuron_type": "Conflicting"}
    for ds in ready_datasets:
        row = values[ds]
        coll_row[f"{ds}_stats"] = (
            f"{to_float(row.get('coll_prop')):.2f} / "
            f"{to_float(row.get('coll_grad')):.2f} / "
            f"{to_float(row.get('coll_total')):.2f}"
        )
        conf_row[f"{ds}_stats"] = (
            f"{to_float(row.get('conf_prop')):.2f} / "
            f"{to_float(row.get('conf_grad')):.2f} / "
            f"{to_float(row.get('conf_total')):.2f}"
        )

    table = {
        "title": "Table 2 (SmolLM): Collaborative vs Conflicting",
        "columns": columns,
        "rows": [coll_row, conf_row],
    }
    return table, ready_datasets, pending_datasets


def build_cnl_table3(settings, jobs_rows, model_tag):
    datasets = settings.get("datasets", ["arc_c", "csqa", "medqa", "mmlu"])
    use_freeze_vals = settings.get("use_freeze", [1, 0])
    target_epoch = int(settings.get("table3_target_epoch", 25))
    method_labels = settings.get("table3_method_labels", {})
    # Paper-style method naming for Table 3: CNL vs FT baseline.
    default_method_labels = {1: "CNL", 0: "FT"}

    by_pair = {}
    for row in jobs_rows:
        ds = row.get("dataset")
        uf = to_int(row.get("use_freeze"), default=None)
        if ds is None or uf is None:
            continue
        by_pair[(str(ds), uf)] = row

    ready_datasets = []
    pending_datasets = []
    for ds in datasets:
        complete = True
        for uf in use_freeze_vals:
            row = by_pair.get((ds, int(uf)))
            if not row:
                complete = False
                break
            if to_int(row.get("epoch"), default=0) < target_epoch:
                complete = False
                break
        if complete:
            ready_datasets.append(ds)
        else:
            pending_datasets.append(ds)

    if not ready_datasets:
        return None, ready_datasets, pending_datasets

    columns = [
        {"key": "model", "label": "MODEL"},
        {"key": "method", "label": "METHOD"},
    ]
    for ds in ready_datasets:
        columns.append({"key": f"{ds}_learned", "label": f"{dataset_title(ds)} LEARNED"})
        columns.append({"key": f"{ds}_forgot", "label": f"{dataset_title(ds)} FORGOT"})

    rows = []
    for idx, uf_raw in enumerate(use_freeze_vals):
        uf = int(uf_raw)
        method = method_labels.get(str(uf), default_method_labels.get(uf, f"use_freeze={uf}"))
        row = {"model": model_tag if idx == 0 else "", "method": method}
        for ds in ready_datasets:
            stats = by_pair.get((ds, uf), {})
            wrong_total = count_nonempty_lines(Path(f"data/{ds}_wrong_{model_tag}.jsonl"))
            correct_total = count_nonempty_lines(Path(f"data/{ds}_correct_{model_tag}.jsonl"))
            learned = to_int(stats.get("wrong_to_correct"), default=0)
            forgot = to_int(stats.get("correct_to_wrong"), default=0)
            row[f"{ds}_learned"] = f"{learned} ({pct_text(learned, wrong_total)})"
            row[f"{ds}_forgot"] = f"{forgot} ({pct_text(forgot, correct_total)})"
        rows.append(row)

    table = {
        "title": "Table 3 (SmolLM): LEARNED vs FORGOT",
        "columns": columns,
        "rows": rows,
    }
    return table, ready_datasets, pending_datasets


def build_cnl_table4(settings, model_tag):
    table4_csv = settings.get("table4_csv_path", "results/smollm_table4.csv")
    csv_path = Path(table4_csv)
    if csv_path.exists():
        rows = read_csv_rows(csv_path, limit=200)
        if rows:
            return {
                "title": "Table 4 (SmolLM): Ablation",
                "columns": [{"key": k, "label": k} for k in rows[0].keys()],
                "rows": rows,
            }, f"Loaded from {table4_csv}"

    placeholder = {
        "title": "Table 4 (SmolLM): Ablation",
        "columns": [
            {"key": "model", "label": "MODEL"},
            {"key": "status", "label": "Status"},
        ],
        "rows": [
            {
                "model": "SmolLM-360M",
                "status": (
                    "Not available yet: Table 4 ablation artifacts are missing "
                    "(lambda ablation runs are required)."
                ),
            }
        ],
    }
    return placeholder, f"Missing file: {table4_csv}"


def resolve_public_link(link_cfg):
    if not link_cfg:
        return None

    label = link_cfg.get("label", "Public Link")
    url = link_cfg.get("url", "")
    source = "static"
    source_path = ""

    if link_cfg.get("source") == "cloudflared_log":
        source = "cloudflared_log"
        regex = re.compile(link_cfg.get("regex", r"https://[-a-z0-9]+\.trycloudflare\.com"))
        logs = sorted(Path(".").glob(link_cfg.get("glob", "logs/cloudflared_*.out")), reverse=True)
        for log_path in logs:
            txt = log_path.read_text(encoding="utf-8", errors="replace")
            matches = regex.findall(txt)
            if matches:
                url = matches[-1]
                source_path = str(log_path)
                break

    if link_cfg.get("url_cmd"):
        source = "command"
        cmd_out = run_cmd(link_cfg["url_cmd"], shell=True)
        if not cmd_out.startswith("ERROR:"):
            url = cmd_out.splitlines()[-1].strip()

    if not url:
        return None

    http_code = run_cmd(
        ["curl", "-s", "-o", "/dev/null", "-m", "8", "-w", "%{http_code}", f"{url}/"],
        timeout=10,
    )
    healthy_codes = set(str(x) for x in link_cfg.get("healthy_http_codes", ["200", "401", "403"]))
    if http_code in healthy_codes:
        health = "auth_required" if http_code == "401" else "healthy"
    else:
        health = "unhealthy"

    return {
        "label": label,
        "url": url,
        "health": health,
        "http_code": http_code,
        "source": source,
        "source_path": source_path,
    }


def collect_cnl(project):
    settings = project.get("settings", {})
    model_tag = settings.get("model_tag", "SmolLM2-360M-Instruct")
    datasets = settings.get("datasets", ["arc_c", "csqa", "medqa", "mmlu"])
    use_freeze_vals = settings.get("use_freeze", [1, 0])
    job_cfg = settings.get("job", {})

    matrix_cfg = {
        "title": "Training Jobs",
        "dimensions": {"dataset": datasets, "use_freeze": use_freeze_vals},
        "log_path_template": job_cfg.get("log_path_template", "logs/train_{dataset}_usefreeze{use_freeze}.log"),
        "summary_csv_template": job_cfg.get(
            "summary_csv_template",
            "zero_ckpts/{dataset}_{model_tag}_lr1e-7_usefreeze{use_freeze}/summary.csv",
        ),
        "time_path_template": job_cfg.get("time_path_template", "logs/train_{dataset}_usefreeze{use_freeze}.time"),
        "metrics": ["train_avg_loss", "wrong_to_correct", "correct_to_wrong"],
        "error_markers": job_cfg.get("error_markers", []),
        "done_markers": job_cfg.get("done_markers", []),
        "epoch_regex": job_cfg.get("epoch_regex", r"^===== Epoch ([0-9]+) =====$"),
        "columns": [
            {"key": "dataset", "label": "dataset"},
            {"key": "use_freeze", "label": "use_freeze"},
            {"key": "status", "label": "status"},
            {"key": "epoch", "label": "epoch"},
            {"key": "train_avg_loss", "label": "train_avg_loss"},
            {"key": "wrong_to_correct", "label": "wrong_to_correct"},
            {"key": "correct_to_wrong", "label": "correct_to_wrong"},
            {"key": "elapsed", "label": "elapsed"},
        ],
    }
    jobs_table = collect_matrix_table(matrix_cfg, {"model_tag": model_tag})

    cards = [{"label": "Model Tag", "value": model_tag}]
    texts = []
    tables = [jobs_table]
    links = []

    table1, t1_ready, t1_pending, t1_neg_totals = build_cnl_table1(settings, jobs_table.get("rows", []), model_tag)
    if table1:
        tables.append(table1)
    t1_ready_text = ", ".join(dataset_title(ds) for ds in t1_ready) or "none"
    t1_pending_text = ", ".join(dataset_title(ds) for ds in t1_pending) or "none"
    neg_details = ", ".join(f"{dataset_title(ds)}={t1_neg_totals.get(ds, 0)}" for ds in t1_ready) or "none"
    texts.append(
        {
            "title": "Table 1 Status",
            "content": (
                f"Rendered datasets: {t1_ready_text}\n"
                f"Pending datasets: {t1_pending_text}\n"
                f"Negative grad-dot totals: {neg_details}"
            ),
        }
    )

    table2, t2_ready, t2_pending = build_cnl_table2(settings, model_tag)
    if table2:
        tables.append(table2)
    t2_ready_text = ", ".join(dataset_title(ds) for ds in t2_ready) or "none"
    t2_pending_text = ", ".join(dataset_title(ds) for ds in t2_pending) or "none"
    texts.append(
        {
            "title": "Table 2 Status",
            "content": f"Rendered datasets: {t2_ready_text}\nPending datasets: {t2_pending_text}",
        }
    )

    table3, ready_datasets, pending_datasets = build_cnl_table3(settings, jobs_table.get("rows", []), model_tag)
    if table3:
        tables.append(table3)
    target_epoch = int(settings.get("table3_target_epoch", 25))
    ready_text = ", ".join(dataset_title(ds) for ds in ready_datasets) or "none"
    pending_text = ", ".join(dataset_title(ds) for ds in pending_datasets) or "none"
    texts.append(
        {
            "title": "Table 3 Status",
            "content": (
                f"Rendered datasets: {ready_text}\n"
                f"Pending datasets (waiting for epoch {target_epoch} on all methods): {pending_text}"
            ),
        }
    )

    table4, table4_status = build_cnl_table4(settings, model_tag)
    if table4:
        tables.append(table4)
    texts.append({"title": "Table 4 Status", "content": table4_status})

    tables.append(collect_gpu_table())

    merged_summary_path = settings.get("merged_summary_path", "")
    if merged_summary_path:
        merged_rows = read_csv_rows(merged_summary_path, limit=200)
        if merged_rows:
            tables.append(
                {
                    "title": "Merged Summary",
                    "columns": [{"key": k, "label": k} for k in merged_rows[0].keys()],
                    "rows": merged_rows,
                }
            )
            cards.append({"label": "Merged Summary", "value": merged_summary_path})

    coord_cfg = settings.get("coordinator", {})
    coord_glob = coord_cfg.get("log_glob", "logs/full_smollm_baselines_*.out")
    coord_logs = sorted(Path(".").glob(coord_glob), reverse=True)
    if coord_logs:
        coord_path = coord_logs[0]
        coord_tail_lines = int(coord_cfg.get("tail_lines", GLOBAL_MAX_LOG_LINES))
        coord_tail = "\n".join(tail_lines(coord_path, coord_tail_lines))
        texts.append({"title": f"Coordinator Log Tail ({coord_path})", "content": coord_tail})
        cards.append({"label": "Coordinator Log", "value": str(coord_path)})
    else:
        texts.append({"title": "Coordinator Log Tail", "content": "No coordinator log found."})

    for panel in settings.get("process_panels", []):
        title = panel.get("title", "Process Panel")
        cmd = panel.get("cmd", "")
        content = run_cmd(cmd, shell=True) if cmd else ""
        texts.append({"title": title, "content": content})

    public_link_cfg = settings.get("public_link")
    if public_link_cfg:
        link = resolve_public_link(public_link_cfg)
        if link:
            links.append(link)
    for link_cfg in settings.get("links", []):
        link = resolve_public_link(link_cfg)
        if link:
            links.append(link)

    return {"cards": cards, "tables": tables, "texts": texts, "links": links}


def collect_generic(project):
    settings = project.get("settings", {})
    shared_values = settings.get("shared_values", {})
    cards = []
    tables = []
    texts = []
    links = []

    if settings.get("include_gpu", True):
        tables.append(collect_gpu_table())

    for card_cfg in settings.get("command_cards", []):
        label = card_cfg.get("label", "Card")
        cmd = card_cfg.get("cmd", "")
        value = run_cmd(cmd, shell=True) if cmd else ""
        cards.append({"label": label, "value": value})

    for matrix_cfg in settings.get("matrix_jobs", []):
        tables.append(collect_matrix_table(matrix_cfg, shared_values))

    for table_cfg in settings.get("csv_tables", []):
        title = table_cfg.get("title", "CSV Table")
        path_value = fmt(table_cfg.get("path", ""), shared_values)
        mode = table_cfg.get("mode", "all")
        limit = int(table_cfg.get("limit", 100))
        tables.append(collect_csv_table(title, path_value, mode=mode, limit=limit))

    for panel in settings.get("log_panels", []):
        title = panel.get("title", "Log Tail")
        path_value = fmt(panel.get("path", ""), shared_values)
        lines = int(panel.get("lines", GLOBAL_MAX_LOG_LINES))
        content = "\n".join(tail_lines(path_value, lines))
        texts.append({"title": title, "content": content})

    for panel in settings.get("command_panels", []):
        title = panel.get("title", "Command Output")
        cmd = panel.get("cmd", "")
        content = run_cmd(cmd, shell=True) if cmd else ""
        texts.append({"title": title, "content": content})

    for link_cfg in settings.get("links", []):
        link = resolve_public_link(link_cfg)
        if link:
            links.append(link)

    return {"cards": cards, "tables": tables, "texts": texts, "links": links}


def collect_project():
    now = datetime.now(timezone.utc).isoformat()
    host = run_cmd(["hostname"])
    cwd = str(Path(".").resolve())
    project_id = PROJECT.get("id", "project")
    project_name = PROJECT.get("name", project_id)
    collector = PROJECT.get("collector", "generic")

    base_cards = [
        {"label": "Remote Host", "value": host},
        {"label": "Remote CWD", "value": cwd},
        {"label": "Collected At (UTC)", "value": now},
    ]

    try:
        if collector == "cnl":
            data = collect_cnl(PROJECT)
        elif collector == "generic":
            data = collect_generic(PROJECT)
        else:
            raise ValueError(f"Unsupported collector: {collector}")
        return {
            "project_id": project_id,
            "project_name": project_name,
            "collector": collector,
            "status": "ok",
            "error": "",
            "collected_at_utc": now,
            "remote_host": host,
            "remote_cwd": cwd,
            "cards": base_cards + data.get("cards", []),
            "tables": data.get("tables", []),
            "texts": data.get("texts", []),
            "links": data.get("links", []),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "project_id": project_id,
            "project_name": project_name,
            "collector": collector,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "collected_at_utc": now,
            "remote_host": host,
            "remote_cwd": cwd,
            "cards": base_cards,
            "tables": [],
            "texts": [{"title": "Collection Error", "content": f"{type(exc).__name__}: {exc}"}],
            "links": [],
        }


result = collect_project()
print(__START_MARKER__)
print(json.dumps(result, indent=2, ensure_ascii=True))
print(__END_MARKER__)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/remote_status_projects.json")
    parser.add_argument("--project-ids", default="")
    parser.add_argument("--output-json", default="data/run_status.json")
    parser.add_argument("--output-html", default="cnl-dashboard.html")
    parser.add_argument("--html-json-path", default="data/run_status.json")
    parser.add_argument("--dashboard-title", default="")
    parser.add_argument("--max-log-lines", type=int, default=80)
    parser.add_argument("--compat-cnl-json", default="data/cnl_status.json")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        (re.compile(r"(ghp_[A-Za-z0-9_]+)"), "***"),
        (re.compile(r"(github_pat_[A-Za-z0-9_]+)"), "***"),
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


def is_link_visible(link: dict[str, Any]) -> bool:
    url = str(link.get("url", "")).strip()
    health = str(link.get("health", "")).strip().lower()
    if not url:
        return False
    return health in {"healthy", "auth_required"}


def build_probe(project: dict[str, Any], max_log_lines: int) -> str:
    payload = {"project": project, "max_log_lines": max_log_lines}
    probe = REMOTE_PROBE_TEMPLATE.replace("__PAYLOAD_JSON__", repr(json.dumps(payload)))
    probe = probe.replace("__START_MARKER__", repr(START_MARKER))
    probe = probe.replace("__END_MARKER__", repr(END_MARKER))
    return probe


def collect_project_status(project: dict[str, Any], ssh_options: list[str], max_log_lines: int) -> dict[str, Any]:
    remote = project.get("remote", {})
    host = str(remote.get("host", "")).strip()
    workdir = str(remote.get("workdir", "")).strip()
    remote_python = str(remote.get("python", "python3")).strip()
    if not host or not workdir:
        raise RuntimeError(f"Project {project.get('id', '<unknown>')} missing remote host/workdir")

    probe = build_probe(project=project, max_log_lines=max_log_lines)

    with tempfile.TemporaryDirectory(prefix="run_status_probe_") as tmpdir:
        probe_path = Path(tmpdir) / "probe.py"
        out_path = Path(tmpdir) / "stdout.txt"
        err_path = Path(tmpdir) / "stderr.txt"
        probe_path.write_text(probe, encoding="utf-8")

        ssh_opts = " ".join(shlex.quote(opt) for opt in ssh_options)
        ssh_host = shlex.quote(host)
        remote_cmd = f"cd {shlex.quote(workdir)} && {shlex.quote(remote_python)} -"
        shell_cmd = (
            f"cat {shlex.quote(str(probe_path))} | "
            f"ssh {ssh_opts} {ssh_host} "
            f"\"{remote_cmd}\" "
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
                f"SSH probe failed (code {proc.returncode}) for {project.get('id', host)}:\n{stderr.strip()}"
            )
        return extract_marked_json(stdout)


def choose_projects(config: dict[str, Any], project_ids_raw: str) -> list[dict[str, Any]]:
    projects = config.get("projects", [])
    selected_ids = [x.strip() for x in project_ids_raw.split(",") if x.strip()]
    if selected_ids:
        wanted = set(selected_ids)
        selected = [p for p in projects if p.get("id") in wanted]
        missing = [pid for pid in selected_ids if pid not in {p.get("id") for p in selected}]
        if missing:
            raise RuntimeError(f"Unknown project ids in --project-ids: {', '.join(missing)}")
        return selected
    return [p for p in projects if p.get("enabled", True)]


def write_dashboard_html(path: Path, json_path: str, title: str) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f2f8fc;
      --ink: #112538;
      --muted: #516a80;
      --card: #ffffff;
      --line: #d4e1eb;
      --ok: #13652b;
      --warn: #936200;
      --bad: #a71f32;
    }}
    body {{
      margin: 0;
      background: linear-gradient(160deg, #eaf3fa 0%, #f8fbff 62%);
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
    }}
    .wrap {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: 0.3px;
    }}
    .muted {{
      color: var(--muted);
      font-size: 14px;
      margin: 0 0 14px;
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 16px;
    }}
    select {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 8px 10px;
      color: var(--ink);
      min-width: 260px;
      font-size: 14px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 12px;
      border: 1px solid transparent;
    }}
    .status-ok {{
      color: var(--ok);
      background: #e6f5ea;
      border-color: #bfe3c8;
    }}
    .status-error {{
      color: var(--bad);
      background: #fdecef;
      border-color: #f7c9d2;
    }}
    .grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      margin: 12px 0 16px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 8px 20px rgba(23, 50, 76, 0.07);
    }}
    .card .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }}
    .card .value {{
      margin-top: 5px;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 14px;
      line-height: 1.35;
    }}
    .section {{
      margin-bottom: 16px;
    }}
    .section h2 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      text-align: left;
      font-size: 13px;
      vertical-align: top;
    }}
    th {{
      background: #e8f1f8;
      font-size: 12px;
      letter-spacing: 0.3px;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    .status-running {{ color: var(--warn); font-weight: 700; }}
    .status-done {{ color: var(--ok); font-weight: 700; }}
    .status-error {{ color: var(--bad); font-weight: 700; }}
    .status-pending {{ color: #607689; font-weight: 700; }}
    pre {{
      margin: 0;
      padding: 12px;
      border-radius: 10px;
      background: #0f172a;
      color: #d9e4f0;
      font-size: 12px;
      line-height: 1.35;
      overflow: auto;
      max-height: 330px;
    }}
    a {{
      color: #005cc5;
      text-decoration: none;
      font-weight: 700;
    }}
    @media (max-width: 800px) {{
      .wrap {{
        padding: 16px;
      }}
      h1 {{
        font-size: 24px;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{title}</h1>
    <p class="muted" id="meta">Loading dashboard snapshot...</p>

    <div class="toolbar">
      <label for="projectSelect"><strong>Project</strong></label>
      <select id="projectSelect"></select>
      <span id="projectStatus" class="badge status-ok">-</span>
    </div>

    <div id="projectError" class="section"></div>
    <div id="cards" class="grid"></div>
    <div id="links" class="section"></div>
    <div id="tables"></div>
    <div id="texts"></div>
  </div>

  <script>
    const JSON_PATH = {json_path!r};
    let payload = null;

    function esc(value) {{
      return String(value ?? "").replace(/[&<>]/g, (ch) => {{
        if (ch === "&") return "&amp;";
        if (ch === "<") return "&lt;";
        return "&gt;";
      }});
    }}

    function renderTable(table) {{
      const cols = table.columns || [];
      const rows = table.rows || [];
      let html = `<div class="section"><h2>${{esc(table.title || "Table")}}</h2>`;
      html += "<table><tr>";
      for (const c of cols) {{
        html += `<th>${{esc(c.label || c.key || "")}}</th>`;
      }}
      html += "</tr>";
      for (const row of rows) {{
        html += "<tr>";
        for (const c of cols) {{
          const key = c.key || "";
          const value = row[key] ?? "";
          const cls = key === "status" ? `status-${{String(value).toLowerCase()}}` : "";
          html += `<td class="${{esc(cls)}}">${{esc(value)}}</td>`;
        }}
        html += "</tr>";
      }}
      html += "</table></div>";
      return html;
    }}

    function renderProject(project) {{
      const statusEl = document.getElementById("projectStatus");
      statusEl.textContent = project.status || "unknown";
      statusEl.className = `badge status-${{project.status === "error" ? "error" : "ok"}}`;

      const errEl = document.getElementById("projectError");
      if (project.status === "error") {{
        errEl.innerHTML = `<div class="card"><strong>Collection Error</strong><div class="value">${{esc(project.error || "unknown")}}</div></div>`;
      }} else {{
        errEl.innerHTML = "";
      }}

      const cards = project.cards || [];
      document.getElementById("cards").innerHTML = cards.map((card) => `
        <div class="card">
          <div class="label">${{esc(card.label || "")}}</div>
          <div class="value">${{esc(card.value || "")}}</div>
        </div>
      `).join("");

      const links = project.links || [];
      const linksEl = document.getElementById("links");
      if (links.length > 0) {{
        linksEl.innerHTML = `
          <h2>Verified Links</h2>
          <div class="card">
            ${{
              links.map((link) => `<div><a href="${{esc(link.url)}}" target="_blank" rel="noopener noreferrer">${{esc(link.label || link.url)}}</a> <span class="muted">(health: ${{esc(link.health || "")}}, http: ${{esc(link.http_code || "")}})</span></div>`).join("")
            }}
          </div>
        `;
      }} else {{
        linksEl.innerHTML = "";
      }}

      const tables = project.tables || [];
      document.getElementById("tables").innerHTML = tables.map((t) => renderTable(t)).join("");

      const texts = project.texts || [];
      document.getElementById("texts").innerHTML = texts.map((block) => `
        <div class="section">
          <h2>${{esc(block.title || "Text")}}</h2>
          <pre>${{esc(block.content || "")}}</pre>
        </div>
      `).join("");
    }}

    function projectById(id) {{
      const projects = payload.projects || [];
      return projects.find((p) => p.project_id === id) || projects[0] || null;
    }}

    function bindSelect(defaultId) {{
      const select = document.getElementById("projectSelect");
      const projects = payload.projects || [];
      const previousValue = select.value || "";
      select.innerHTML = projects.map((p) => `
        <option value="${{esc(p.project_id)}}">${{esc(p.project_name || p.project_id)}}</option>
      `).join("");
      const preferred = previousValue || defaultId;
      const exists = projects.some((p) => p.project_id === preferred);
      select.value = exists ? preferred : (projects[0] ? projects[0].project_id : "");
      const current = projectById(select.value);
      if (current) renderProject(current);
      select.onchange = () => {{
        const chosen = projectById(select.value);
        if (chosen) renderProject(chosen);
      }};
    }}

    async function load() {{
      const res = await fetch(`${{JSON_PATH}}?t=${{Date.now()}}`, {{ cache: "no-store" }});
      payload = await res.json();
      document.getElementById("meta").textContent = `Snapshot generated: ${{payload.generated_at_utc || "-"}} | Auto-refresh: 15s`;
      bindSelect(payload.default_project_id || "");
    }}

    async function tick() {{
      try {{
        await load();
      }} catch (err) {{
        document.getElementById("meta").textContent = `Load error: ${{err}}`;
      }}
    }}

    tick();
    setInterval(tick, 15000);
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_compat_cnl_json(path: Path, payload: dict[str, Any]) -> None:
    project = None
    target_id = payload.get("default_project_id")
    for item in payload.get("projects", []):
        if item.get("project_id") == target_id:
            project = item
            break
    if not project and payload.get("projects"):
        project = payload["projects"][0]
    if not project:
        return

    compat: dict[str, Any] = {
        "collected_at_utc": project.get("collected_at_utc", ""),
        "host": project.get("remote_host", ""),
        "cwd": project.get("remote_cwd", ""),
        "status": project.get("status", ""),
        "project_id": project.get("project_id", ""),
        "project_name": project.get("project_name", ""),
        "cards": project.get("cards", []),
        "tables": project.get("tables", []),
        "texts": project.get("texts", []),
        "links": project.get("links", []),
    }

    jobs_table = next((t for t in project.get("tables", []) if t.get("title") == "Training Jobs"), None)
    gpu_table = next((t for t in project.get("tables", []) if t.get("title") == "GPU"), None)
    if jobs_table:
        compat["jobs"] = jobs_table.get("rows", [])
    if gpu_table:
        compat["gpu"] = gpu_table.get("rows", [])
    if project.get("links"):
        compat["public_dashboard_url"] = project["links"][0].get("url", "")
        compat["public_dashboard_http_noauth"] = project["links"][0].get("http_code", "")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compat, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    defaults = config.get("defaults", {})
    ssh_options = list(defaults.get("ssh_options", ["-o", "BatchMode=yes"]))

    projects = choose_projects(config=config, project_ids_raw=args.project_ids)
    if not projects:
        raise RuntimeError("No projects selected.")

    out_projects: list[dict[str, Any]] = []
    for project in projects:
        try:
            data = collect_project_status(
                project=project,
                ssh_options=ssh_options,
                max_log_lines=args.max_log_lines,
            )
            data = sanitize_obj(data)
            data["links"] = [x for x in data.get("links", []) if is_link_visible(x)]
            out_projects.append(data)
        except Exception as exc:  # noqa: BLE001
            out_projects.append(
                {
                    "project_id": project.get("id", "unknown"),
                    "project_name": project.get("name", project.get("id", "unknown")),
                    "collector": project.get("collector", ""),
                    "status": "error",
                    "error": str(exc),
                    "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                    "remote_host": project.get("remote", {}).get("host", ""),
                    "remote_cwd": project.get("remote", {}).get("workdir", ""),
                    "cards": [],
                    "tables": [],
                    "texts": [{"title": "Collection Error", "content": str(exc)}],
                    "links": [],
                }
            )

    default_project_id = defaults.get("default_project_id")
    if args.project_ids:
        requested = [x.strip() for x in args.project_ids.split(",") if x.strip()]
        if requested:
            default_project_id = requested[0]
    if default_project_id not in {p.get("project_id") for p in out_projects} and out_projects:
        default_project_id = out_projects[0].get("project_id")

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dashboard_title": args.dashboard_title or defaults.get("dashboard_title", "Remote Training Status Dashboard"),
        "default_project_id": default_project_id,
        "projects": out_projects,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    if args.output_html:
        output_html = Path(args.output_html)
        write_dashboard_html(
            path=output_html,
            json_path=args.html_json_path,
            title=payload["dashboard_title"],
        )

    compat_path = str(args.compat_cnl_json).strip()
    if compat_path:
        write_compat_cnl_json(Path(compat_path), payload=payload)

    print(f"Wrote {output_json}")
    if args.output_html:
        print(f"Wrote {args.output_html}")
    if compat_path:
        print(f"Wrote {compat_path}")
    print(f"Projects collected: {len(out_projects)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
