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
        "arc_c": "ARC-C",
        "csqa": "CSQA",
        "mmlu": "MMLU",
        "medqa": "MedQA",
    }
    return names.get(dataset, dataset)


def order_datasets_for_paper(settings, datasets):
    preferred = settings.get("paper_dataset_order", ["mmlu", "medqa", "arc_c", "csqa"])
    ordered = [ds for ds in preferred if ds in datasets]
    ordered.extend(ds for ds in datasets if ds not in ordered)
    return ordered


def pct_text(numerator, denominator):
    if denominator <= 0:
        return "0.00%"
    return f"{(100.0 * numerator / denominator):.2f}%"


def parse_count_pct(value):
    text = str(value or "").strip()
    m = re.match(r"^\s*([0-9]+)\s*\(([0-9.]+)%\)\s*$", text)
    if not m:
        return 0, 0.0
    return int(m.group(1)), float(m.group(2))


def parse_pct_only(value):
    text = str(value or "").strip().rstrip("%")
    try:
        return float(text)
    except Exception:  # noqa: BLE001
        return 0.0


def build_simple_paper_comparison_text(table1, table2, table3):
    avg_ft_learn = None
    avg_cnl_learn = None
    avg_ft_forgot = None
    avg_cnl_forgot = None
    forgot_reduction = None
    learn_change = None
    has_t3 = False

    if table3 and table3.get("rows"):
        rows = table3.get("rows", [])
        ft_row = next((r for r in rows if str(r.get("method", "")).upper() == "FT"), None)
        cnl_row = next((r for r in rows if str(r.get("method", "")).upper() == "CNL"), None)
        ds_order = table3.get("dataset_order", [])
        if ft_row and cnl_row and ds_order:
            ft_learn = []
            cnl_learn = []
            ft_forgot = []
            cnl_forgot = []
            for ds in ds_order:
                _, ft_lp = parse_count_pct(ft_row.get(f"{ds}_learned", "0 (0.00%)"))
                _, cnl_lp = parse_count_pct(cnl_row.get(f"{ds}_learned", "0 (0.00%)"))
                _, ft_fp = parse_count_pct(ft_row.get(f"{ds}_forgot", "0 (0.00%)"))
                _, cnl_fp = parse_count_pct(cnl_row.get(f"{ds}_forgot", "0 (0.00%)"))
                ft_learn.append(ft_lp)
                cnl_learn.append(cnl_lp)
                ft_forgot.append(ft_fp)
                cnl_forgot.append(cnl_fp)

            avg_ft_learn = sum(ft_learn) / len(ft_learn)
            avg_cnl_learn = sum(cnl_learn) / len(cnl_learn)
            avg_ft_forgot = sum(ft_forgot) / len(ft_forgot)
            avg_cnl_forgot = sum(cnl_forgot) / len(cnl_forgot)
            forgot_reduction = (avg_ft_forgot - avg_cnl_forgot) / avg_ft_forgot * 100.0 if avg_ft_forgot > 0 else 0.0
            learn_change = (avg_cnl_learn - avg_ft_learn) / avg_ft_learn * 100.0 if avg_ft_learn > 0 else 0.0
            has_t3 = True

    stronger_sim = None
    if table1 and table1.get("rows"):
        row = table1["rows"][0]
        ds_order = table1.get("dataset_order", [])
        sim_count = 0
        for ds in ds_order:
            _, dissim_pct = parse_count_pct(row.get(f"{ds}_dissimilar", "0 (0.00%)"))
            _, sim_pct = parse_count_pct(row.get(f"{ds}_similar", "0 (0.00%)"))
            if sim_pct > dissim_pct:
                sim_count += 1
        stronger_sim = (sim_count, len(ds_order))

    conf_dom = None
    totals_normalized = False
    if table2 and table2.get("rows"):
        ds_order = table2.get("dataset_order", [])
        by_metric = {str(r.get("metric", "")).upper(): r for r in table2.get("rows", [])}
        prop_row = by_metric.get("PROP", {})
        conf_count = 0
        for ds in ds_order:
            coll = parse_pct_only(prop_row.get(f"{ds}_coll", "0"))
            conf = parse_pct_only(prop_row.get(f"{ds}_conf", "0"))
            if conf > coll:
                conf_count += 1
        conf_dom = (conf_count, len(ds_order))

        totals_normalized = True
        total_row = by_metric.get("TOTAL", {})
        for ds in ds_order:
            try:
                if abs(float(str(total_row.get(f"{ds}_total", "0"))) - 100.0) > 1e-6:
                    totals_normalized = False
                    break
            except Exception:  # noqa: BLE001
                totals_normalized = False
                break

    lines = []
    lines.append("Quick read (simple language):")
    lines.append("")

    lines.append("1) What these results mean in practical and conceptual terms:")
    if has_t3:
        lines.append(
            f"- Practical: Forgetting drops a lot with CNL (FT {avg_ft_forgot:.2f}% -> CNL {avg_cnl_forgot:.2f}%)."
        )
        lines.append(
            f"- Practical: New learning is also much lower with CNL (FT {avg_ft_learn:.2f}% -> CNL {avg_cnl_learn:.2f}%)."
        )
        lines.append(
            "- Conceptual: On this small model, CNL behaves like a strong stability control. "
            "It protects old knowledge, but it also limits plasticity."
        )
    else:
        lines.append("- Table 3 is incomplete, so practical meaning is not stable yet.")
    if stronger_sim:
        lines.append(
            f"- Mechanism hint from Table 1: Sim has more forgetting in {stronger_sim[0]}/{stronger_sim[1]} datasets."
        )
    if conf_dom:
        lines.append(
            f"- Mechanism hint from Table 2: Conflicting neurons are more common in {conf_dom[0]}/{conf_dom[1]} datasets."
        )

    lines.append("")
    lines.append("2) How this compares to the original paper (larger models):")
    lines.append("- Direction matches: CNL still reduces forgetting compared with FT.")
    if has_t3 and forgot_reduction is not None and learn_change is not None:
        lines.append(
            f"- In this SmolLM run, forgetting is about {forgot_reduction:.1f}% lower with CNL, "
            f"but learning is about {abs(learn_change):.1f}% {'lower' if learn_change < 0 else 'higher'}."
        )
    lines.append(
        "- Difference from paper trend: the large-model paper shows strong forgetting control "
        "without such a large learning drop."
    )
    lines.append("- So this is a partial reproduction, not a full quantitative match.")

    lines.append("")
    lines.append("3) Are trends consistent with the original findings? If not, why might they differ?")
    lines.append("- Partly consistent: CNL < FT on forgetting.")
    lines.append("- Partly inconsistent: CNL learning is much lower here than the paper suggests.")
    lines.append("- Likely reasons:")
    lines.append("- Small model has less spare capacity, so stability controls can block learning more strongly.")
    lines.append("- Same training recipe from bigger models may not be optimal at this scale.")
    lines.append("- One model size and one config are not enough to average out run variance.")
    lines.append("- Table 2 values are normalized in this pipeline, while the paper reports signed sums.")

    lines.append("")
    lines.append("4) What we can and cannot conclude (scalability, robustness, mechanism):")
    lines.append("- Can conclude (robustness): CNL robustly reduces forgetting versus FT in this run.")
    lines.append("- Can conclude (mechanism, directional): conflict-heavy structure still appears (Table 2), but weaker.")
    lines.append("- Cannot conclude (scalability): we cannot claim large-model behavior from one small-model setting.")
    lines.append("- Cannot conclude (mechanism, strict): we need exact metric parity and multi-seed checks.")
    lines.append("- Cannot conclude (final quality): this is a strong diagnostic result, not the final endpoint.")

    return "\n".join(lines)


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

    ordered_datasets = order_datasets_for_paper(settings, ready_datasets)

    row = {"model": settings.get("paper_model_name", "SmolLM 360M")}
    for ds in ordered_datasets:
        row[f"{ds}_dissimilar"] = values[ds]["dissimilar"]
        row[f"{ds}_similar"] = values[ds]["similar"]

    table = {
        "title": "Table 1",
        "paper_table": "table1",
        "caption": settings.get(
            "table1_caption",
            (
                "Relationship between gradient similarity and catastrophic forgetting. "
                "Results are reported as Number of forgotten questions (Percentage). "
                "Samples with negative gradient similarity are ranked by magnitude: "
                "top 1/3 are Sim, bottom 1/3 are Dissim."
            ),
        ),
        "dataset_order": ordered_datasets,
        "columns": [{"key": "model", "label": "MODEL"}],
        "rows": [row],
    }
    return table, ordered_datasets, pending_datasets, neg_totals


def build_cnl_table1_evolution(settings, jobs_rows, model_tag):
    datasets = settings.get("datasets", ["arc_c", "csqa", "medqa", "mmlu"])
    ft_use_freeze = int(settings.get("table1_ft_use_freeze", 0))
    target_epoch = int(settings.get("table1_target_epoch", 25))
    palette = settings.get(
        "table1_evolution_palette",
        {"mmlu": "#1772b7", "medqa": "#17a673", "arc_c": "#d46b08", "csqa": "#8d4ecf"},
    )

    by_pair = {}
    for row in jobs_rows:
        ds = str(row.get("dataset", ""))
        uf = to_int(row.get("use_freeze"), default=None)
        if ds and uf is not None:
            by_pair[(ds, uf)] = row

    series = []
    ready_datasets = []
    pending_datasets = []
    max_epoch_global = 0

    for ds in datasets:
        ft_row = by_pair.get((ds, ft_use_freeze))
        ft_epoch = to_int((ft_row or {}).get("epoch"), default=0)
        if ft_epoch < 1:
            pending_datasets.append(ds)
            continue

        distance_jsonl = Path(f"distance/{ds}_{model_tag}/correct_with_grad_dot.jsonl")
        jsonl_dir = Path(f"zero_ckpts/{ds}_{model_tag}_lr1e-7_usefreeze{ft_use_freeze}/jsonl")
        if not distance_jsonl.exists() or not jsonl_dir.exists():
            pending_datasets.append(ds)
            continue

        dist_rows = read_jsonl_rows(distance_jsonl)
        neg_rows = [r for r in dist_rows if to_float(r.get("grad_dot"), default=0.0) < 0.0]
        neg_rows.sort(key=lambda r: abs(to_float(r.get("grad_dot"), default=0.0)), reverse=True)
        n_neg = len(neg_rows)
        third = n_neg // 3
        if n_neg <= 0 or third <= 0:
            pending_datasets.append(ds)
            continue

        sim_rows = neg_rows[:third]
        dissim_rows = neg_rows[-third:]
        sim_questions = [str(r.get("question", "")) for r in sim_rows]
        dissim_questions = [str(r.get("question", "")) for r in dissim_rows]
        sim_set = set(sim_questions)
        dissim_set = set(dissim_questions)
        grad_map = {str(r.get("question", "")): to_float(r.get("grad_dot"), default=0.0) for r in neg_rows}
        sim_total_abs_grad = sum(abs(grad_map.get(q, 0.0)) for q in sim_questions)
        dissim_total_abs_grad = sum(abs(grad_map.get(q, 0.0)) for q in dissim_questions)
        sim_sorted_by_abs = sorted(sim_questions, key=lambda q: abs(grad_map.get(q, 0.0)), reverse=True)
        dissim_sorted_by_abs = sorted(dissim_questions, key=lambda q: abs(grad_map.get(q, 0.0)), reverse=True)

        def build_rank_map(sorted_questions):
            size = len(sorted_questions)
            if size <= 1:
                return {q: 100.0 for q in sorted_questions}
            return {q: 100.0 * (1.0 - (idx / (size - 1))) for idx, q in enumerate(sorted_questions)}

        sim_rank_map = build_rank_map(sim_sorted_by_abs)
        dissim_rank_map = build_rank_map(dissim_sorted_by_abs)

        epoch_files = []
        for p in jsonl_dir.glob("infer_correct_ep*.jsonl"):
            m = re.search(r"infer_correct_ep([0-9]+)\.jsonl$", p.name)
            if not m:
                continue
            ep = to_int(m.group(1), default=0)
            if ep > 0:
                epoch_files.append((ep, p))
        epoch_files.sort(key=lambda x: x[0])

        sim_points = []
        dissim_points = []
        prev_sim_forgot_set = set()
        prev_dissim_forgot_set = set()
        for ep, path_obj in epoch_files:
            rows = read_jsonl_rows(path_obj)
            forgot_questions = set()
            for r in rows:
                q = str(r.get("question", ""))
                if str(r.get("label", "")) != str(r.get("predict_label", "")):
                    forgot_questions.add(q)

            sim_forgot_q = [q for q in sim_questions if q in forgot_questions]
            dissim_forgot_q = [q for q in dissim_questions if q in forgot_questions]
            sim_forgot_set = set(sim_forgot_q)
            dissim_forgot_set = set(dissim_forgot_q)
            sim_new_forgot_q = [q for q in sim_questions if q in sim_forgot_set and q not in prev_sim_forgot_set]
            dissim_new_forgot_q = [q for q in dissim_questions if q in dissim_forgot_set and q not in prev_dissim_forgot_set]
            sim_count = len(sim_forgot_q)
            dissim_count = len(dissim_forgot_q)
            sim_pct = 100.0 * sim_count / n_neg if n_neg > 0 else 0.0
            dissim_pct = 100.0 * dissim_count / n_neg if n_neg > 0 else 0.0

            sim_avg_grad = None
            dissim_avg_grad = None
            sim_abs_grad_forgot = sum(abs(grad_map.get(q, 0.0)) for q in sim_forgot_q)
            dissim_abs_grad_forgot = sum(abs(grad_map.get(q, 0.0)) for q in dissim_forgot_q)
            sim_abs_mass_pct = 100.0 * sim_abs_grad_forgot / sim_total_abs_grad if sim_total_abs_grad > 0 else 0.0
            dissim_abs_mass_pct = (
                100.0 * dissim_abs_grad_forgot / dissim_total_abs_grad if dissim_total_abs_grad > 0 else 0.0
            )
            sim_new_abs_grad = sum(abs(grad_map.get(q, 0.0)) for q in sim_new_forgot_q)
            dissim_new_abs_grad = sum(abs(grad_map.get(q, 0.0)) for q in dissim_new_forgot_q)
            sim_new_abs_mass_pct = 100.0 * sim_new_abs_grad / sim_total_abs_grad if sim_total_abs_grad > 0 else 0.0
            dissim_new_abs_mass_pct = (
                100.0 * dissim_new_abs_grad / dissim_total_abs_grad if dissim_total_abs_grad > 0 else 0.0
            )
            sim_cum_rank_pct = (
                sum(sim_rank_map.get(q, 0.0) for q in sim_forgot_q) / sim_count if sim_count > 0 else None
            )
            dissim_cum_rank_pct = (
                sum(dissim_rank_map.get(q, 0.0) for q in dissim_forgot_q) / dissim_count if dissim_count > 0 else None
            )
            sim_new_rank_pct = (
                sum(sim_rank_map.get(q, 0.0) for q in sim_new_forgot_q) / len(sim_new_forgot_q)
                if len(sim_new_forgot_q) > 0
                else None
            )
            dissim_new_rank_pct = (
                sum(dissim_rank_map.get(q, 0.0) for q in dissim_new_forgot_q) / len(dissim_new_forgot_q)
                if len(dissim_new_forgot_q) > 0
                else None
            )
            if sim_count > 0:
                sim_avg_grad = sum(grad_map.get(q, 0.0) for q in sim_forgot_q) / sim_count
            if dissim_count > 0:
                dissim_avg_grad = sum(grad_map.get(q, 0.0) for q in dissim_forgot_q) / dissim_count

            sim_points.append(
                {
                    "epoch": ep,
                    "forgot_count": sim_count,
                    "forgot_pct": round(sim_pct, 4),
                    "avg_grad": None if sim_avg_grad is None else round(sim_avg_grad, 8),
                    "forgot_abs_grad_sum": round(sim_abs_grad_forgot, 8),
                    "class_abs_grad_total": round(sim_total_abs_grad, 8),
                    "abs_grad_mass_pct": round(sim_abs_mass_pct, 4),
                    "new_forgot_count": len(sim_new_forgot_q),
                    "new_abs_grad_mass_pct": round(sim_new_abs_mass_pct, 4),
                    "cum_rank_pct": None if sim_cum_rank_pct is None else round(sim_cum_rank_pct, 4),
                    "new_rank_pct": None if sim_new_rank_pct is None else round(sim_new_rank_pct, 4),
                }
            )
            dissim_points.append(
                {
                    "epoch": ep,
                    "forgot_count": dissim_count,
                    "forgot_pct": round(dissim_pct, 4),
                    "avg_grad": None if dissim_avg_grad is None else round(dissim_avg_grad, 8),
                    "forgot_abs_grad_sum": round(dissim_abs_grad_forgot, 8),
                    "class_abs_grad_total": round(dissim_total_abs_grad, 8),
                    "abs_grad_mass_pct": round(dissim_abs_mass_pct, 4),
                    "new_forgot_count": len(dissim_new_forgot_q),
                    "new_abs_grad_mass_pct": round(dissim_new_abs_mass_pct, 4),
                    "cum_rank_pct": None if dissim_cum_rank_pct is None else round(dissim_cum_rank_pct, 4),
                    "new_rank_pct": None if dissim_new_rank_pct is None else round(dissim_new_rank_pct, 4),
                }
            )
            prev_sim_forgot_set = sim_forgot_set
            prev_dissim_forgot_set = dissim_forgot_set
            if ep > max_epoch_global:
                max_epoch_global = ep

        if not sim_points or not dissim_points:
            pending_datasets.append(ds)
            continue

        ready_datasets.append(ds)
        series.append(
            {
                "dataset": ds,
                "dataset_label": dataset_title(ds),
                "color": str(palette.get(ds, "")),
                "negative_total": n_neg,
                "sim_pool_size": len(sim_set),
                "dissim_pool_size": len(dissim_set),
                "sim": sim_points,
                "dissim": dissim_points,
            }
        )

    if not series:
        ready_text = ", ".join(dataset_title(ds) for ds in ready_datasets) if ready_datasets else "none"
        pending_text = ", ".join(dataset_title(ds) for ds in pending_datasets) if pending_datasets else "none"
        status = (
            f"Rendered datasets: {ready_text}\n"
            f"Pending datasets: {pending_text}\n"
            "Max epoch observed: 0"
        )
        return None, ready_datasets, pending_datasets, status

    ordered = order_datasets_for_paper(settings, [s["dataset"] for s in series])
    idx = {ds: i for i, ds in enumerate(ordered)}
    series.sort(key=lambda s: idx.get(s.get("dataset", ""), 999))

    ready_text = ", ".join(dataset_title(ds) for ds in ordered) if ordered else "none"
    pending_text = ", ".join(dataset_title(ds) for ds in pending_datasets) if pending_datasets else "none"

    table = {
        "title": "Table 1 Evolution",
        "viz": "table1_evolution",
        "caption": settings.get(
            "table1_evolution_caption",
            (
                "Evolution of Table 1 over training checkpoints. X-axis is checkpoint point index and "
                "Y-axis is forgotten-question percentage; each dataset shows SIM and DISSIM curves."
            ),
        ),
        "dataset_order": ordered,
        "target_epoch": target_epoch,
        "max_epoch": max_epoch_global,
        "series": series,
    }
    status = (
        f"Rendered datasets: {ready_text}\n"
        f"Pending datasets: {pending_text}\n"
        f"Max epoch observed: {max_epoch_global}"
    )
    return table, ordered, pending_datasets, status


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

    ordered_datasets = order_datasets_for_paper(settings, ready_datasets)
    prop_row = {"metric": "PROP"}
    grad_row = {"metric": "GRAD"}
    total_row = {"metric": "TOTAL"}
    for ds in ordered_datasets:
        row = values[ds]
        coll_prop = to_float(row.get("coll_prop"))
        conf_prop = to_float(row.get("conf_prop"))
        coll_grad = to_float(row.get("coll_grad"))
        conf_grad = to_float(row.get("conf_grad"))
        coll_total = to_float(row.get("coll_total"))
        conf_total = to_float(row.get("conf_total"))

        prop_row[f"{ds}_coll"] = f"{coll_prop:.1f}%"
        prop_row[f"{ds}_conf"] = f"{conf_prop:.1f}%"
        grad_row[f"{ds}_coll"] = f"{coll_grad:+.2f}"
        grad_row[f"{ds}_conf"] = f"{conf_grad:+.2f}"
        total_row[f"{ds}_total"] = f"{(coll_total + conf_total):.2f}"

    table = {
        "title": "Table 2",
        "paper_table": "table2",
        "caption": settings.get(
            "table2_caption",
            (
                "Distribution of collaborative neurons (COLL) and conflicting neurons (CONF). "
                "PROP denotes neuron-type proportion. GRAD denotes gradient similarity sum for that type. "
                "TOTAL denotes the gradient similarity sum over all neurons."
            ),
        ),
        "dataset_order": ordered_datasets,
        "model": settings.get("paper_model_name", "SmolLM 360M"),
        "rows": [prop_row, grad_row, total_row],
        "columns": [{"key": "metric", "label": "Metric"}],
    }
    return table, ordered_datasets, pending_datasets


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

    ordered_datasets = order_datasets_for_paper(settings, ready_datasets)
    use_freeze_ints = [int(x) for x in use_freeze_vals]
    preferred_method_order = settings.get("table3_method_order", [0, 1])
    method_order = []
    for uf_raw in preferred_method_order:
        uf = int(uf_raw)
        if uf in use_freeze_ints and uf not in method_order:
            method_order.append(uf)
    for uf in use_freeze_ints:
        if uf not in method_order:
            method_order.append(uf)

    columns = [{"key": "model", "label": "MODEL"}, {"key": "method", "label": "METHOD"}]
    for ds in ordered_datasets:
        columns.append({"key": f"{ds}_learned", "label": f"{dataset_title(ds)} LEARNED"})
        columns.append({"key": f"{ds}_forgot", "label": f"{dataset_title(ds)} FORGOT"})

    rows = []
    for uf in method_order:
        method = method_labels.get(str(uf), default_method_labels.get(uf, f"use_freeze={uf}"))
        row = {"model": settings.get("paper_model_name", "SmolLM 360M"), "method": method}
        for ds in ordered_datasets:
            stats = by_pair.get((ds, uf), {})
            wrong_total = count_nonempty_lines(Path(f"data/{ds}_wrong_{model_tag}.jsonl"))
            correct_total = count_nonempty_lines(Path(f"data/{ds}_correct_{model_tag}.jsonl"))
            learned = to_int(stats.get("wrong_to_correct"), default=0)
            forgot = to_int(stats.get("correct_to_wrong"), default=0)
            row[f"{ds}_learned"] = f"{learned} ({pct_text(learned, wrong_total)})"
            row[f"{ds}_forgot"] = f"{forgot} ({pct_text(forgot, correct_total)})"
        rows.append(row)

    table = {
        "title": "Table 3",
        "paper_table": "table3",
        "caption": settings.get(
            "table3_caption",
            (
                "Effectiveness of knowledge injection using FT and CNL. Results are reported "
                "as Number of questions (Percentage). FT tends to forget previously mastered "
                "knowledge, while CNL induces negligible catastrophic forgetting."
            ),
        ),
        "dataset_order": ordered_datasets,
        "model": settings.get("paper_model_name", "SmolLM 360M"),
        "columns": columns,
        "rows": rows,
    }
    return table, ordered_datasets, pending_datasets


def build_cnl_table4(settings, model_tag):
    datasets = settings.get("datasets", ["arc_c", "csqa", "medqa", "mmlu"])
    ordered_datasets = order_datasets_for_paper(settings, datasets)
    target_epoch = int(settings.get("table4_target_epoch", 25))
    lr = str(settings.get("table4_lr", "1e-7"))
    optimizer_rows = settings.get(
        "table4_optimizer_rows",
        [
            {"key": "sgd", "label": "SGD"},
            {"key": "momentum", "label": "Momentum"},
            {"key": "adam", "label": "Adam"},
            {"key": "adamw", "label": "AdamW"},
        ],
    )
    method_rows = settings.get("table4_method_rows", [{"use_freeze": 0, "label": "FT"}, {"use_freeze": 1, "label": "CNL"}])

    wrong_totals = {}
    correct_totals = {}
    for ds in ordered_datasets:
        wrong_totals[ds] = count_nonempty_lines(Path(f"data/{ds}_wrong_{model_tag}.jsonl"))
        correct_totals[ds] = count_nonempty_lines(Path(f"data/{ds}_correct_{model_tag}.jsonl"))

    def summary_path(ds, opt_key, use_freeze):
        uf = int(use_freeze)
        if opt_key == "sgd":
            return Path(f"zero_ckpts/{ds}_{model_tag}_lr{lr}_usefreeze{uf}/summary.csv")
        if opt_key == "momentum":
            return Path(f"optimizers/moment_{ds}_{model_tag}_lr{lr}_usefreeze{uf}/summary.csv")
        if opt_key == "adam":
            return Path(f"optimizers/adam_{ds}_{model_tag}_lr{lr}_usefreeze{uf}/summary.csv")
        if opt_key == "adamw":
            return Path(f"optimizers/adamw_{ds}_{model_tag}_lr{lr}_usefreeze{uf}/summary.csv")
        return Path("")

    rows = []
    ready = 0
    pending = 0
    pending_items = []
    for opt_cfg in optimizer_rows:
        opt_key = str(opt_cfg.get("key", ""))
        opt_label = str(opt_cfg.get("label", opt_key))
        for method_cfg in method_rows:
            uf = int(method_cfg.get("use_freeze", 0))
            method_label = str(method_cfg.get("label", f"use_freeze={uf}"))
            row = {"optimizer": opt_label, "method": method_label}
            for ds in ordered_datasets:
                s_path = summary_path(ds, opt_key, uf)
                last = read_last_csv_row(s_path) if s_path and str(s_path) else {}
                epoch = to_int(last.get("epoch"), default=0)
                if not last or epoch < target_epoch:
                    row[f"{ds}_learned"] = "PENDING"
                    row[f"{ds}_forgot"] = "PENDING"
                    pending += 1
                    pending_items.append(f"{opt_label}/{method_label}/{dataset_title(ds)}")
                    continue

                learned = to_int(last.get("wrong_to_correct"), default=0)
                forgot = to_int(last.get("correct_to_wrong"), default=0)
                row[f"{ds}_learned"] = f"{learned} ({pct_text(learned, wrong_totals.get(ds, 0))})"
                row[f"{ds}_forgot"] = f"{forgot} ({pct_text(forgot, correct_totals.get(ds, 0))})"
                ready += 1
            rows.append(row)

    if not rows:
        placeholder = {
            "title": "Table 4",
            "columns": [{"key": "status", "label": "Status"}],
            "rows": [{"status": "No ablation rows configured."}],
        }
        return placeholder, "No optimizer/method rows configured."

    total = ready + pending
    pending_preview = ", ".join(pending_items[:8]) if pending_items else "none"
    if pending > 8:
        pending_preview += f", +{pending - 8} more"

    table = {
        "title": "Table 4",
        "paper_table": "table4",
        "caption": settings.get(
            "table4_caption",
            (
                "Impact of different optimizers on catastrophic forgetting. Results are "
                "reported as Number of questions (Percentage)."
            ),
        ),
        "dataset_order": ordered_datasets,
        "model": settings.get("paper_model_name", "SmolLM 360M"),
        "target_epoch": target_epoch,
        "columns": [{"key": "optimizer", "label": "OPTIMIZER"}, {"key": "method", "label": "METHOD"}],
        "rows": rows,
    }
    status = (
        f"Completed entries: {ready}/{total} (target epoch {target_epoch}).\n"
        f"Pending entries: {pending}\n"
        f"Pending preview: {pending_preview}"
    )
    return table, status


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

    table1_evo, t1e_ready, t1e_pending, t1e_status = build_cnl_table1_evolution(
        settings, jobs_table.get("rows", []), model_tag
    )
    if table1_evo:
        tables.append(table1_evo)
    texts.append(
        {
            "title": "Table 1 Evolution Status",
            "content": t1e_status,
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
    texts.append(
        {
            "title": "Paper Comparison Appendix (Simple)",
            "content": build_simple_paper_comparison_text(table1, table2, table3),
        }
    )

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
    primary_host = str(remote.get("host", "")).strip()
    host_candidates_raw = remote.get("host_candidates", [])
    host_candidates = []
    if primary_host:
        host_candidates.append(primary_host)
    if isinstance(host_candidates_raw, list):
        for item in host_candidates_raw:
            value = str(item).strip()
            if value and value not in host_candidates:
                host_candidates.append(value)

    workdir = str(remote.get("workdir", "")).strip()
    remote_python = str(remote.get("python", "python3")).strip()
    if not host_candidates or not workdir:
        raise RuntimeError(f"Project {project.get('id', '<unknown>')} missing remote host/workdir")

    probe = build_probe(project=project, max_log_lines=max_log_lines)

    with tempfile.TemporaryDirectory(prefix="run_status_probe_") as tmpdir:
        probe_path = Path(tmpdir) / "probe.py"
        out_path = Path(tmpdir) / "stdout.txt"
        err_path = Path(tmpdir) / "stderr.txt"
        probe_path.write_text(probe, encoding="utf-8")

        ssh_opts = " ".join(shlex.quote(opt) for opt in ssh_options)
        remote_cmd = f"cd {shlex.quote(workdir)} && {shlex.quote(remote_python)} -"
        errors = []
        for host in host_candidates:
            ssh_host = shlex.quote(host)
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
            if proc.returncode == 0:
                out = extract_marked_json(stdout)
                out["ssh_endpoint"] = host
                return out

            err_text = stderr.strip() or "unknown ssh error"
            errors.append(f"{host}: code {proc.returncode}: {err_text}")

        raise RuntimeError(
            f"SSH probe failed for {project.get('id', '<unknown>')} across {len(host_candidates)} host(s):\n"
            + "\n".join(errors)
        )


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
    .status-warn {{
      color: var(--warn);
      background: #fff4dc;
      border-color: #f0d8a7;
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
    .appendix-shell {{
      position: relative;
      overflow: hidden;
      border: 1px solid #c9d7e5;
      border-radius: 16px;
      background:
        radial-gradient(circle at 6% 12%, rgba(209, 233, 251, 0.75), rgba(209, 233, 251, 0) 45%),
        radial-gradient(circle at 92% 0%, rgba(255, 238, 210, 0.72), rgba(255, 238, 210, 0) 40%),
        linear-gradient(155deg, #ffffff 0%, #f6fbff 52%, #fffaf3 100%);
      box-shadow: 0 18px 34px rgba(16, 50, 81, 0.13);
      padding: 18px;
    }}
    .appendix-shell::after {{
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background: linear-gradient(110deg, rgba(255, 255, 255, 0.28), rgba(255, 255, 255, 0));
    }}
    .appendix-kicker {{
      position: relative;
      z-index: 1;
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 11px;
      letter-spacing: 0.9px;
      text-transform: uppercase;
      color: #114061;
      background: rgba(213, 232, 247, 0.78);
      border: 1px solid rgba(108, 151, 184, 0.3);
      font-weight: 700;
    }}
    .appendix-title {{
      position: relative;
      z-index: 1;
      margin: 10px 0 2px;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      font-size: 30px;
      line-height: 1.1;
      letter-spacing: 0.2px;
      color: #0f2f48;
    }}
    .appendix-subtitle {{
      position: relative;
      z-index: 1;
      margin: 0 0 16px;
      color: #36566f;
      font-size: 14px;
      line-height: 1.45;
    }}
    .appendix-grid {{
      position: relative;
      z-index: 1;
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }}
    .appendix-panel {{
      border-radius: 12px;
      border: 1px solid #d4e2ee;
      background: rgba(255, 255, 255, 0.78);
      backdrop-filter: blur(1px);
      padding: 12px;
    }}
    .appendix-panel h3 {{
      margin: 0 0 8px;
      font-size: 15px;
      color: #0f3049;
      letter-spacing: 0.2px;
    }}
    .appendix-panel p {{
      margin: 0 0 8px;
      color: #24465f;
      font-size: 13px;
      line-height: 1.45;
    }}
    .appendix-list {{
      margin: 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 7px;
    }}
    .appendix-list li {{
      position: relative;
      padding-left: 16px;
      color: #1f415a;
      font-size: 13px;
      line-height: 1.42;
    }}
    .appendix-list li::before {{
      content: "";
      position: absolute;
      left: 0;
      top: 0.56em;
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #2d7fb5;
      box-shadow: 0 0 0 3px rgba(45, 127, 181, 0.12);
    }}
    .appendix-panel.tone-good {{
      border-color: #c7e3d1;
      background: linear-gradient(150deg, rgba(241, 252, 245, 0.95), rgba(255, 255, 255, 0.84));
    }}
    .appendix-panel.tone-good h3 {{
      color: #0f4a2b;
    }}
    .appendix-panel.tone-good .appendix-list li::before {{
      background: #2f9553;
      box-shadow: 0 0 0 3px rgba(47, 149, 83, 0.13);
    }}
    .appendix-panel.tone-caution {{
      border-color: #ead8bf;
      background: linear-gradient(150deg, rgba(255, 249, 237, 0.95), rgba(255, 255, 255, 0.84));
    }}
    .appendix-panel.tone-caution h3 {{
      color: #6b4f14;
    }}
    .appendix-panel.tone-caution .appendix-list li::before {{
      background: #b98721;
      box-shadow: 0 0 0 3px rgba(185, 135, 33, 0.15);
    }}
    .paper-wrap {{
      background: #fff;
      border: 1px solid #cfd7de;
      border-radius: 10px;
      padding: 10px 12px 12px;
      overflow-x: auto;
    }}
    .paper-caption {{
      margin: 0 0 8px;
      font-family: "Times New Roman", Georgia, serif;
      font-size: 18px;
      line-height: 1.15;
      color: #111;
    }}
    .paper-caption .paper-title {{
      font-weight: 700;
    }}
    .paper-table {{
      width: 100%;
      border-collapse: collapse;
      border: none;
      border-radius: 0;
      font-family: "Times New Roman", Georgia, serif;
      background: #fff;
    }}
    .paper-table thead tr:first-child th {{
      border-top: 2px solid #333;
    }}
    .paper-table thead th {{
      background: #fff;
      border-bottom: 1px solid #555;
      color: #111;
      font-size: 13px;
      font-variant: small-caps;
      letter-spacing: 0.2px;
      padding: 5px 6px 3px;
      text-align: center;
      white-space: nowrap;
    }}
    .paper-table td {{
      border-bottom: 1px solid #666;
      color: #111;
      font-size: 14px;
      line-height: 1.02;
      padding: 4px 7px;
      text-align: center;
      vertical-align: middle;
      white-space: nowrap;
    }}
    .paper-table tbody tr:last-child td {{
      border-bottom: 2px solid #333;
    }}
    .paper-table .left {{
      text-align: left;
    }}
    .paper-table .italic {{
      font-style: italic;
    }}
    .paper-table .sub {{
      display: block;
      font-size: 0.92em;
      margin-top: 1px;
    }}
    .paper-sep td {{
      border-top: 1px solid #555;
    }}
    .evo-wrap {{
      background: linear-gradient(165deg, #ffffff 0%, #f8fbff 100%);
      border: 1px solid #c8d8e8;
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 14px 30px rgba(22, 47, 77, 0.08);
    }}
    .evo-caption {{
      margin: 0 0 10px;
      color: #1f425f;
      font-size: 13.5px;
      line-height: 1.4;
    }}
    .evo-panels {{
      display: grid;
      gap: 18px;
      grid-template-columns: 1fr;
    }}
    .evo-panel {{
      border: 1px solid #d0deec;
      border-radius: 12px;
      background: linear-gradient(160deg, #ffffff 0%, #f4f9ff 100%);
      padding: 16px;
    }}
    .evo-panel h3 {{
      margin: 0 0 12px;
      color: #0f334d;
      font-size: 22px;
      letter-spacing: 0.15px;
      font-weight: 700;
      line-height: 1.2;
    }}
    .evo-chart {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid #d4e0ec;
      border-radius: 10px;
      background: #ffffff;
    }}
    .evo-legend {{
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      align-items: center;
      justify-content: center;
    }}
    .evo-legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: #234a66;
      font-size: 15px;
      white-space: nowrap;
      letter-spacing: 0.1px;
    }}
    .evo-chip {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
      border: 1px solid rgba(17, 37, 56, 0.2);
    }}
    .evo-line-sample {{
      width: 34px;
      border-top: 3px solid #243b53;
      position: relative;
      top: -1px;
    }}
    .evo-line-sample.dissim {{
      border-top-style: dashed;
    }}
    .evo-marker-sample {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      background: #243b53;
    }}
    .evo-marker-sample.square {{
      border-radius: 2px;
    }}
    .evo-meta {{
      margin-top: 10px;
      color: #3d5f79;
      font-size: 14px;
      line-height: 1.45;
      text-align: center;
    }}
    .evo-grad-deck {{
      display: grid;
      gap: 12px;
    }}
    .evo-grad-main {{
      display: block;
    }}
    .evo-grad-note {{
      margin: 0;
      color: #2f526d;
      font-size: 16px;
      line-height: 1.45;
      text-align: center;
    }}
    .evo-small-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 12px;
      margin-top: 4px;
    }}
    .evo-mini-panel {{
      border: 1px solid #d4e0ec;
      border-radius: 10px;
      background: #ffffff;
      padding: 10px;
    }}
    .evo-mini-title {{
      margin: 0 0 6px;
      color: #173c58;
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.1px;
    }}
    .evo-mini-chart {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid #e4edf5;
      border-radius: 8px;
      background: #ffffff;
    }}
    .evo-mini-note {{
      margin: 2px 0 0;
      color: #48657d;
      font-size: 12px;
      text-align: center;
    }}
    @media (max-width: 800px) {{
      .wrap {{
        padding: 16px;
      }}
      h1 {{
        font-size: 24px;
      }}
      .paper-caption {{
        font-size: 16px;
      }}
      .paper-table thead th {{
        font-size: 17px;
      }}
      .paper-table td {{
        font-size: 16px;
      }}
      .appendix-title {{
        font-size: 24px;
      }}
      .appendix-shell {{
        padding: 14px;
      }}
      .appendix-grid {{
        grid-template-columns: 1fr;
      }}
      .evo-panel h3 {{
        font-size: 18px;
      }}
      .evo-legend-item {{
        font-size: 13px;
      }}
      .evo-grad-note {{
        font-size: 14px;
      }}
      .evo-mini-title {{
        font-size: 13px;
      }}
      .evo-mini-note {{
        font-size: 11px;
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
    <div id="appendix" class="section"></div>
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

    function datasetLabel(dataset) {{
      const names = {{
        mmlu: "MMLU",
        medqa: "MedQA",
        arc_c: "ARC-C",
        csqa: "CSQA",
      }};
      return names[dataset] || dataset || "";
    }}

    function splitMainSub(rawValue) {{
      const text = String(rawValue ?? "").trim();
      const m = text.match(/^(.+?)\\s*\\(([^)]*)\\)$/);
      if (!m) {{
        return {{ main: text, sub: "" }};
      }}
      return {{ main: m[1].trim(), sub: `(${{m[2].trim()}})` }};
    }}

    function renderMainSubCell(rawValue) {{
      const parts = splitMainSub(rawValue);
      if (!parts.sub) {{
        return esc(parts.main);
      }}
      return `${{esc(parts.main)}}<span class="sub">${{esc(parts.sub)}}</span>`;
    }}

    function paperCaption(table) {{
      const title = table.title || "Table";
      const caption = table.caption || "";
      if (!caption) {{
        return `<p class="paper-caption"><span class="paper-title">${{esc(title)}}.</span></p>`;
      }}
      const prefix = `${{title}}:`;
      let text = caption;
      if (text.startsWith(prefix)) {{
        text = text.slice(prefix.length).trim();
      }}
      return `<p class="paper-caption"><span class="paper-title">${{esc(prefix)}}</span> ${{esc(text)}}</p>`;
    }}

    function renderPaperTable1(table) {{
      const datasets = table.dataset_order || [];
      const rows = table.rows || [];
      const row = rows[0] || {{}};

      let html = `<div class="section paper-wrap">${{paperCaption(table)}}`;
      html += `<table class="paper-table"><thead><tr>`;
      html += `<th class="left" rowspan="2">MODEL</th>`;
      for (const ds of datasets) {{
        html += `<th colspan="2">${{esc(datasetLabel(ds))}}</th>`;
      }}
      html += `</tr><tr>`;
      for (const _ds of datasets) {{
        html += `<th>DISSIM</th><th>SIM</th>`;
      }}
      html += `</tr></thead><tbody><tr>`;
      html += `<td class="left italic">${{esc(row.model || table.model || "")}}</td>`;
      for (const ds of datasets) {{
        html += `<td>${{renderMainSubCell(row[`${{ds}}_dissimilar`])}}</td>`;
        html += `<td>${{renderMainSubCell(row[`${{ds}}_similar`])}}</td>`;
      }}
      html += `</tr></tbody></table></div>`;
      return html;
    }}

    function renderPaperTable2(table) {{
      const datasets = table.dataset_order || [];
      const rows = table.rows || [];
      const byMetric = {{}};
      for (const row of rows) {{
        byMetric[String(row.metric || "").toUpperCase()] = row;
      }}

      const model = table.model || "";
      const prop = byMetric.PROP || {{}};
      const grad = byMetric.GRAD || {{}};
      const total = byMetric.TOTAL || {{}};

      let html = `<div class="section paper-wrap">${{paperCaption(table)}}`;
      html += `<table class="paper-table"><thead><tr>`;
      html += `<th class="left" rowspan="2">MODEL</th>`;
      html += `<th class="left" rowspan="2">METRIC</th>`;
      for (const ds of datasets) {{
        html += `<th colspan="2">${{esc(datasetLabel(ds))}}</th>`;
      }}
      html += `</tr><tr>`;
      for (const _ds of datasets) {{
        html += `<th>COLL</th><th>CONF</th>`;
      }}
      html += `</tr></thead><tbody>`;

      html += `<tr>`;
      html += `<td class="left italic" rowspan="3">${{esc(model)}}</td>`;
      html += `<td class="left">PROP</td>`;
      for (const ds of datasets) {{
        html += `<td>${{esc(prop[`${{ds}}_coll`] || "")}}</td>`;
        html += `<td>${{esc(prop[`${{ds}}_conf`] || "")}}</td>`;
      }}
      html += `</tr>`;

      html += `<tr>`;
      html += `<td class="left">GRAD</td>`;
      for (const ds of datasets) {{
        html += `<td>${{esc(grad[`${{ds}}_coll`] || "")}}</td>`;
        html += `<td>${{esc(grad[`${{ds}}_conf`] || "")}}</td>`;
      }}
      html += `</tr>`;

      html += `<tr>`;
      html += `<td class="left">TOTAL</td>`;
      for (const ds of datasets) {{
        html += `<td colspan="2">${{esc(total[`${{ds}}_total`] || "")}}</td>`;
      }}
      html += `</tr>`;

      html += `</tbody></table></div>`;
      return html;
    }}

    function renderPaperTable3(table) {{
      const datasets = table.dataset_order || [];
      const rows = table.rows || [];
      const orderedRows = [];
      const seen = new Set();
      for (const method of ["FT", "CNL"]) {{
        const idx = rows.findIndex((row) => String(row.method || "").toUpperCase() === method);
        if (idx >= 0) {{
          orderedRows.push(rows[idx]);
          seen.add(idx);
        }}
      }}
      rows.forEach((row, idx) => {{
        if (!seen.has(idx)) orderedRows.push(row);
      }});

      const model = table.model || ((orderedRows[0] || {{}}).model || "");
      let html = `<div class="section paper-wrap">${{paperCaption(table)}}`;
      html += `<table class="paper-table"><thead><tr>`;
      html += `<th class="left" rowspan="2">MODEL</th>`;
      html += `<th class="left" rowspan="2">METHOD</th>`;
      for (const ds of datasets) {{
        html += `<th colspan="2">${{esc(datasetLabel(ds))}}</th>`;
      }}
      html += `</tr><tr>`;
      for (const _ds of datasets) {{
        html += `<th>LEARNED</th><th>FORGOT</th>`;
      }}
      html += `</tr></thead><tbody>`;

      orderedRows.forEach((row, idx) => {{
        const sepClass = idx > 0 ? " paper-sep" : "";
        html += `<tr class="${{sepClass.trim()}}">`;
        if (idx === 0) {{
          html += `<td class="left italic" rowspan="${{orderedRows.length}}">${{esc(model)}}</td>`;
        }}
        html += `<td class="left">${{esc(row.method || "")}}</td>`;
        for (const ds of datasets) {{
          html += `<td>${{renderMainSubCell(row[`${{ds}}_learned`])}}</td>`;
          html += `<td>${{renderMainSubCell(row[`${{ds}}_forgot`])}}</td>`;
        }}
        html += `</tr>`;
      }});

      html += `</tbody></table></div>`;
      return html;
    }}

    function renderPaperValue(rawValue) {{
      const text = String(rawValue ?? "");
      if (text.trim().toUpperCase() === "PENDING") {{
        return `<span class="status-pending">PENDING</span>`;
      }}
      if (text.includes("(") && text.includes("%")) {{
        return renderMainSubCell(text);
      }}
      return esc(text);
    }}

    function renderPaperTable4(table) {{
      const datasets = table.dataset_order || [];
      const rows = table.rows || [];
      let html = `<div class="section paper-wrap">${{paperCaption(table)}}`;
      html += `<table class="paper-table"><thead><tr>`;
      html += `<th class="left" rowspan="2">OPTIMIZER</th>`;
      html += `<th class="left" rowspan="2">METHOD</th>`;
      for (const ds of datasets) {{
        html += `<th colspan="2">${{esc(datasetLabel(ds))}}</th>`;
      }}
      html += `</tr><tr>`;
      for (const _ds of datasets) {{
        html += `<th>LEARNED</th><th>FORGOT</th>`;
      }}
      html += `</tr></thead><tbody>`;

      const groupSizes = {{}};
      for (const row of rows) {{
        const key = String(row.optimizer || "");
        groupSizes[key] = (groupSizes[key] || 0) + 1;
      }}

      let prevOpt = null;
      for (let i = 0; i < rows.length; i += 1) {{
        const row = rows[i] || {{}};
        const opt = String(row.optimizer || "");
        const isNewGroup = i === 0 || opt !== prevOpt;
        const sepClass = isNewGroup && i > 0 ? " paper-sep" : "";
        html += `<tr class="${{sepClass.trim()}}">`;
        if (isNewGroup) {{
          html += `<td class="left" rowspan="${{groupSizes[opt] || 1}}">${{esc(opt)}}</td>`;
        }}
        html += `<td class="left">${{esc(row.method || "")}}</td>`;
        for (const ds of datasets) {{
          html += `<td>${{renderPaperValue(row[`${{ds}}_learned`])}}</td>`;
          html += `<td>${{renderPaperValue(row[`${{ds}}_forgot`])}}</td>`;
        }}
        html += `</tr>`;
        prevOpt = opt;
      }}

      html += `</tbody></table></div>`;
      return html;
    }}

    function toFiniteNumber(value, fallback = null) {{
      const num = Number(value);
      return Number.isFinite(num) ? num : fallback;
    }}

    function buildLinearTicks(minValue, maxValue, steps = 6) {{
      if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {{
        return [];
      }}
      if (steps <= 1 || minValue === maxValue) {{
        return [minValue];
      }}
      const ticks = [];
      for (let i = 0; i < steps; i += 1) {{
        ticks.push(minValue + ((maxValue - minValue) * i) / (steps - 1));
      }}
      return ticks;
    }}

    function scaleLinear(value, domainMin, domainMax, rangeMin, rangeMax) {{
      if (domainMax === domainMin) {{
        return (rangeMin + rangeMax) / 2;
      }}
      const ratio = (value - domainMin) / (domainMax - domainMin);
      return rangeMin + ratio * (rangeMax - rangeMin);
    }}

    function classLabel(classKey) {{
      return classKey === "dissim" ? "DISSIM" : "SIM";
    }}

    function classDash(classKey) {{
      return classKey === "dissim" ? "7 5" : "";
    }}

    function collectEvolutionGroups(table) {{
      const groups = [];
      for (const entry of table.series || []) {{
        const dataset = String(entry.dataset || "");
        const datasetName = String(entry.dataset_label || datasetLabel(dataset));
        const color = String(entry.color || "#2f6ea2");
        for (const classKey of ["sim", "dissim"]) {{
          const rawPoints = Array.isArray(entry[classKey]) ? entry[classKey] : [];
          const points = rawPoints
            .map((point) => ({{
              epoch: toFiniteNumber(point.epoch, null),
              forgot_count: toFiniteNumber(point.forgot_count, null),
              forgot_pct: toFiniteNumber(point.forgot_pct, null),
              avg_grad: toFiniteNumber(point.avg_grad, null),
              abs_grad_mass_pct: toFiniteNumber(point.abs_grad_mass_pct, null),
              forgot_abs_grad_sum: toFiniteNumber(point.forgot_abs_grad_sum, null),
              class_abs_grad_total: toFiniteNumber(point.class_abs_grad_total, null),
              new_forgot_count: toFiniteNumber(point.new_forgot_count, null),
              new_abs_grad_mass_pct: toFiniteNumber(point.new_abs_grad_mass_pct, null),
              cum_rank_pct: toFiniteNumber(point.cum_rank_pct, null),
              new_rank_pct: toFiniteNumber(point.new_rank_pct, null),
            }}))
            .filter((point) => point.epoch !== null)
            .sort((a, b) => a.epoch - b.epoch);
          if (points.length > 0) {{
            groups.push({{
              dataset: dataset,
              dataset_label: datasetName,
              color: color,
              class_key: classKey,
              points: points,
            }});
          }}
        }}
      }}
      return groups;
    }}

    function renderEvolutionPrimarySvg(groups, epochMax) {{
      const rows = [];
      const epochSet = new Set();
      for (const group of groups) {{
        const sorted = (group.points || []).slice().sort((a, b) => a.epoch - b.epoch);
        for (const point of sorted) {{
          if (!Number.isFinite(point.forgot_pct)) {{
            continue;
          }}
          const epochInt = Math.round(toFiniteNumber(point.epoch, 0));
          if (!Number.isFinite(epochInt) || epochInt <= 0) {{
            continue;
          }}
          epochSet.add(epochInt);
          rows.push({{
            dataset_label: group.dataset_label,
            color: group.color,
            class_key: group.class_key,
            point_idx: 0,
            epoch: epochInt,
            forgot_pct: point.forgot_pct,
            forgot_count: point.forgot_count,
          }});
        }}
      }}
      if (rows.length === 0) {{
        return '<p class="evo-meta">No forgotten-question points are available yet.</p>';
      }}

      const orderedEpochs = Array.from(epochSet).sort((a, b) => a - b);
      const epochToPoint = new Map();
      for (let i = 0; i < orderedEpochs.length; i += 1) {{
        epochToPoint.set(orderedEpochs[i], i + 1);
      }}
      for (const row of rows) {{
        row.point_idx = epochToPoint.get(row.epoch) || 0;
      }}

      const pctMaxRaw = Math.max(1, ...rows.map((row) => row.forgot_pct));
      const pctMax = Math.max(5, Math.ceil(pctMaxRaw / 5) * 5);
      const pointMax = Math.max(1, orderedEpochs.length);
      const width = 1280;
      const height = 560;
      const margin = {{ top: 26, right: 28, bottom: 78, left: 96 }};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const mapX = (value) => scaleLinear(value, 1, pointMax, margin.left, margin.left + plotWidth);
      const mapY = (value) => scaleLinear(value, 0, pctMax, margin.top + plotHeight, margin.top);
      const xTicksRaw = buildLinearTicks(1, pointMax, Math.min(8, Math.max(4, pointMax)));
      const xTicks = Array.from(new Set(xTicksRaw.map((tick) => Math.round(tick))))
        .filter((tick) => tick >= 1 && tick <= pointMax)
        .sort((a, b) => a - b);
      if (!xTicks.includes(1)) xTicks.unshift(1);
      if (!xTicks.includes(pointMax)) xTicks.push(pointMax);
      const yTicks = buildLinearTicks(0, pctMax, 6);

      let svg = `<svg class="evo-chart" viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Forgotten percentage versus checkpoint point index, grouped by dataset and SIM or DISSIM">`;

      for (const tick of yTicks) {{
        const y = mapY(tick);
        svg += `<line x1="${{margin.left}}" y1="${{y.toFixed(2)}}" x2="${{(margin.left + plotWidth).toFixed(2)}}" y2="${{y.toFixed(2)}}" stroke="#e2ebf4" stroke-width="1.2"></line>`;
        svg += `<text x="${{(margin.left - 10).toFixed(2)}}" y="${{(y + 4.6).toFixed(2)}}" font-size="15" fill="#4e677d" text-anchor="end">${{tick.toFixed(1)}}%</text>`;
      }}
      for (const tick of xTicks) {{
        const x = mapX(tick);
        svg += `<line x1="${{x.toFixed(2)}}" y1="${{margin.top}}" x2="${{x.toFixed(2)}}" y2="${{(margin.top + plotHeight).toFixed(2)}}" stroke="#eef4fa" stroke-width="1.1"></line>`;
        svg += `<text x="${{x.toFixed(2)}}" y="${{(height - 28).toFixed(2)}}" font-size="15" fill="#4e677d" text-anchor="middle">${{tick}}</text>`;
      }}
      svg += `<line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{(margin.top + plotHeight).toFixed(2)}}" stroke="#5e7890" stroke-width="1.8"></line>`;
      svg += `<line x1="${{margin.left}}" y1="${{(margin.top + plotHeight).toFixed(2)}}" x2="${{(margin.left + plotWidth).toFixed(2)}}" y2="${{(margin.top + plotHeight).toFixed(2)}}" stroke="#5e7890" stroke-width="1.8"></line>`;
      svg += `<text x="${{(margin.left + (plotWidth / 2)).toFixed(2)}}" y="${{(height - 8).toFixed(2)}}" font-size="17" fill="#294862" text-anchor="middle">point index</text>`;
      const yLabelAnchor = margin.top + (plotHeight / 2);
      svg += `<text x="24" y="${{yLabelAnchor.toFixed(2)}}" font-size="17" fill="#294862" text-anchor="middle" transform="rotate(-90 24 ${{yLabelAnchor.toFixed(2)}})">forgotten questions (%)</text>`;

      for (const group of groups) {{
        const points = (group.points || [])
          .filter((point) => Number.isFinite(point.forgot_pct) && Number.isFinite(point.epoch))
          .map((point) => {{
            const epochInt = Math.round(toFiniteNumber(point.epoch, 0));
            return {{
              epoch: epochInt,
              point_idx: epochToPoint.get(epochInt) || 0,
              forgot_pct: point.forgot_pct,
              forgot_count: point.forgot_count,
            }};
          }})
          .filter((point) => point.point_idx > 0)
          .sort((a, b) => a.point_idx - b.point_idx);
        if (points.length === 0) {{
          continue;
        }}
        const dash = classDash(group.class_key);
        const dashAttr = dash ? ` stroke-dasharray="${{dash}}"` : "";
        const path = points
          .map((point, idx) => `${{idx === 0 ? "M" : "L"}} ${{mapX(point.point_idx).toFixed(2)}} ${{mapY(point.forgot_pct).toFixed(2)}}`)
          .join(" ");
        svg += `<path d="${{path}}" fill="none" stroke="${{esc(group.color)}}" stroke-width="3.4"${{dashAttr}} stroke-linecap="round" opacity="0.97"></path>`;

        const endPointIdx = points[points.length - 1].point_idx;
        for (const point of points) {{
          const pointIdx = point.point_idx;
          const x = mapX(pointIdx);
          const y = mapY(point.forgot_pct);
          const fade = 0.45 + 0.55 * (Math.max(1, pointIdx) / Math.max(1, pointMax));
          const tooltip = `${{group.dataset_label}} ${{classLabel(group.class_key)}} | point ${{pointIdx}} | epoch ${{point.epoch}} | forgotten ${{point.forgot_count}} (${{point.forgot_pct.toFixed(2)}}%)`;
          if (group.class_key === "dissim") {{
            svg += `<rect x="${{(x - 5.0).toFixed(2)}}" y="${{(y - 5.0).toFixed(2)}}" width="10.0" height="10.0" fill="${{esc(group.color)}}" fill-opacity="${{fade.toFixed(3)}}" stroke="#ffffff" stroke-width="1.1"><title>${{esc(tooltip)}}</title></rect>`;
            if (pointIdx === endPointIdx) {{
              svg += `<rect x="${{(x - 7.0).toFixed(2)}}" y="${{(y - 7.0).toFixed(2)}}" width="14.0" height="14.0" fill="none" stroke="#10212f" stroke-width="1.2"></rect>`;
            }}
          }} else {{
            svg += `<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="5.0" fill="${{esc(group.color)}}" fill-opacity="${{fade.toFixed(3)}}" stroke="#ffffff" stroke-width="1.1"><title>${{esc(tooltip)}}</title></circle>`;
            if (pointIdx === endPointIdx) {{
              svg += `<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="7.2" fill="none" stroke="#10212f" stroke-width="1.2"></circle>`;
            }}
          }}
        }}
      }}

      svg += `</svg>`;
      return svg;
    }}

    function renderEvolutionGradSvg(groups, epochMax) {{
      const classStyle = {{
        sim: {{ label: "SIM", color: "#0E5E5A", band_fill: "rgba(14, 94, 90, 0.18)" }},
        dissim: {{ label: "DISSIM", color: "#A34D1A", band_fill: "rgba(163, 77, 26, 0.15)" }},
      }};
      const datasetMap = new Map();
      const epochSet = new Set();

      function quantile(values, q) {{
        if (!values || values.length === 0) return null;
        const sorted = values.slice().sort((a, b) => a - b);
        if (sorted.length === 1) return sorted[0];
        const pos = (sorted.length - 1) * q;
        const lo = Math.floor(pos);
        const hi = Math.ceil(pos);
        if (lo === hi) return sorted[lo];
        const w = pos - lo;
        return sorted[lo] * (1 - w) + sorted[hi] * w;
      }}

      for (const group of groups) {{
        const classKey = group.class_key;
        if (!(classKey in classStyle)) continue;
        const dsKey = String(group.dataset || group.dataset_label || "");
        if (!datasetMap.has(dsKey)) {{
          datasetMap.set(dsKey, {{
            dataset: dsKey,
            dataset_label: String(group.dataset_label || dsKey),
            color: String(group.color || "#2f6ea2"),
            sim_cum: [],
            sim_new: [],
            dissim_cum: [],
            dissim_new: [],
          }});
        }}
        const bucket = datasetMap.get(dsKey);
        for (const point of group.points || []) {{
          const ep = Math.round(toFiniteNumber(point.epoch, 0));
          if (!Number.isFinite(ep) || ep <= 0) continue;
          epochSet.add(ep);
          const forgotCount = Math.max(0, toFiniteNumber(point.forgot_count, 0));
          const newCount = Math.max(0, toFiniteNumber(point.new_forgot_count, 0));
          if (Number.isFinite(point.cum_rank_pct) && forgotCount > 0) {{
            bucket[`${{classKey}}_cum`].push({{ epoch: ep, value: point.cum_rank_pct, count: forgotCount }});
          }}
          if (Number.isFinite(point.new_rank_pct) && newCount > 0) {{
            bucket[`${{classKey}}_new`].push({{ epoch: ep, value: point.new_rank_pct, count: newCount }});
          }}
        }}
      }}

      const datasets = Array.from(datasetMap.values());
      for (const ds of datasets) {{
        for (const key of ["sim_cum", "sim_new", "dissim_cum", "dissim_new"]) {{
          ds[key].sort((a, b) => a.epoch - b.epoch);
        }}
      }}
      if (datasets.length === 0) {{
        return '<p class="evo-meta">No gradient-rank evolution data are available yet.</p>';
      }}

      function summarizeByClass(key) {{
        const perEpoch = {{}};
        for (const ds of datasets) {{
          for (const point of ds[key] || []) {{
            const ep = point.epoch;
            if (!perEpoch[ep]) perEpoch[ep] = [];
            perEpoch[ep].push(point);
          }}
        }}
        const epochs = Object.keys(perEpoch).map((v) => Number(v)).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
        const out = [];
        for (const ep of epochs) {{
          const pts = perEpoch[ep] || [];
          const vals = pts.map((p) => p.value).filter((v) => Number.isFinite(v));
          if (vals.length === 0) continue;
          let wsum = 0;
          let wtot = 0;
          for (const p of pts) {{
            const w = Math.max(0, toFiniteNumber(p.count, 0));
            wsum += p.value * w;
            wtot += w;
          }}
          out.push({{
            epoch: ep,
            mean: wtot > 0 ? (wsum / wtot) : (vals.reduce((a, b) => a + b, 0) / vals.length),
            q25: quantile(vals, 0.25),
            q75: quantile(vals, 0.75),
            n: vals.length,
            count_sum: wtot,
          }});
        }}
        return out;
      }}

      const simCumStats = summarizeByClass("sim_cum");
      const dissimCumStats = summarizeByClass("dissim_cum");
      const simNewStats = summarizeByClass("sim_new");
      const dissimNewStats = summarizeByClass("dissim_new");

      const allEpochs = Array.from(epochSet).sort((a, b) => a - b);
      const epochCeil = Math.max(1, Math.round(Math.max(epochMax, ...allEpochs)));

      function pathFromSeries(series, mapXFn, mapYFn, valueKey = "mean") {{
        if (!series || series.length === 0) return "";
        let d = "";
        for (let i = 0; i < series.length; i += 1) {{
          const pt = series[i];
          const val = toFiniteNumber(pt[valueKey], null);
          if (!Number.isFinite(val)) continue;
          const x = mapXFn(pt.epoch);
          const y = mapYFn(val);
          if (i === 0 || (pt.epoch - series[i - 1].epoch) > 1) {{
            d += `M ${{x.toFixed(2)}} ${{y.toFixed(2)}} `;
          }} else {{
            d += `L ${{x.toFixed(2)}} ${{y.toFixed(2)}} `;
          }}
        }}
        return d.trim();
      }}

      function bandPaths(stats, mapXFn, mapYFn) {{
        const usable = (stats || [])
          .filter((pt) => Number.isFinite(pt.epoch) && Number.isFinite(pt.q25) && Number.isFinite(pt.q75) && Number(pt.n || 0) >= 2)
          .sort((a, b) => a.epoch - b.epoch);
        if (usable.length < 2) return [];
        const segments = [];
        let seg = [usable[0]];
        for (let i = 1; i < usable.length; i += 1) {{
          const prev = usable[i - 1];
          const cur = usable[i];
          if ((cur.epoch - prev.epoch) <= 1) {{
            seg.push(cur);
          }} else {{
            if (seg.length >= 2) segments.push(seg);
            seg = [cur];
          }}
        }}
        if (seg.length >= 2) segments.push(seg);
        return segments.map((segment) => {{
          const upper = segment.map((pt) => `${{mapXFn(pt.epoch).toFixed(2)}} ${{mapYFn(pt.q75).toFixed(2)}}`);
          const lower = segment.slice().reverse().map((pt) => `${{mapXFn(pt.epoch).toFixed(2)}} ${{mapYFn(pt.q25).toFixed(2)}}`);
          return `M ${{upper[0]}} L ${{upper.slice(1).join(" L ")}} L ${{lower.join(" L ")}} Z`;
        }});
      }}

      const width = 1280;
      const height = 500;
      const margin = {{ top: 24, right: 40, bottom: 70, left: 92 }};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const mapX = (v) => scaleLinear(v, 1, epochCeil, margin.left, margin.left + plotWidth);
      const mapY = (v) => scaleLinear(v, 0, 100, margin.top + plotHeight, margin.top);
      const xTicksRaw = buildLinearTicks(1, epochCeil, Math.min(8, Math.max(4, epochCeil)));
      const xTicks = Array.from(new Set(xTicksRaw.map((t) => Math.round(t)))).filter((t) => t >= 1 && t <= epochCeil).sort((a, b) => a - b);
      if (!xTicks.includes(1)) xTicks.unshift(1);
      if (!xTicks.includes(epochCeil)) xTicks.push(epochCeil);
      const yTicks = [0, 20, 40, 60, 80, 100];

      const maxAggCount = Math.max(
        1,
        ...simNewStats.map((x) => Math.max(0, toFiniteNumber(x.count_sum, 0))),
        ...dissimNewStats.map((x) => Math.max(0, toFiniteNumber(x.count_sum, 0))),
      );
      function aggMarkerRadius(countValue) {{
        const c = Math.max(0, toFiniteNumber(countValue, 0));
        return 3.4 + (Math.sqrt(c) / Math.sqrt(maxAggCount)) * 4.4;
      }}

      let svg = `<svg class="evo-chart" viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Gradient rank evolution aggregate chart">`;
      for (const tick of yTicks) {{
        const y = mapY(tick);
        svg += `<line x1="${{margin.left}}" y1="${{y.toFixed(2)}}" x2="${{(margin.left + plotWidth).toFixed(2)}}" y2="${{y.toFixed(2)}}" stroke="#e4edf6" stroke-width="1.2"></line>`;
        svg += `<text x="${{(margin.left - 10).toFixed(2)}}" y="${{(y + 4.6).toFixed(2)}}" font-size="15" fill="#4e677d" text-anchor="end">${{tick}}</text>`;
      }}
      for (const tick of xTicks) {{
        const x = mapX(tick);
        svg += `<line x1="${{x.toFixed(2)}}" y1="${{margin.top}}" x2="${{x.toFixed(2)}}" y2="${{(margin.top + plotHeight).toFixed(2)}}" stroke="#eef4fa" stroke-width="1.1"></line>`;
        svg += `<text x="${{x.toFixed(2)}}" y="${{(height - 26).toFixed(2)}}" font-size="15" fill="#4e677d" text-anchor="middle">${{tick}}</text>`;
      }}
      svg += `<line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{(margin.top + plotHeight).toFixed(2)}}" stroke="#5e7890" stroke-width="1.8"></line>`;
      svg += `<line x1="${{margin.left}}" y1="${{(margin.top + plotHeight).toFixed(2)}}" x2="${{(margin.left + plotWidth).toFixed(2)}}" y2="${{(margin.top + plotHeight).toFixed(2)}}" stroke="#5e7890" stroke-width="1.8"></line>`;
      svg += `<text x="${{(margin.left + (plotWidth / 2)).toFixed(2)}}" y="${{(height - 6).toFixed(2)}}" font-size="17" fill="#294862" text-anchor="middle">training epoch</text>`;
      const yAnchor = margin.top + (plotHeight / 2);
      svg += `<text x="24" y="${{yAnchor.toFixed(2)}}" font-size="17" fill="#294862" text-anchor="middle" transform="rotate(-90 24 ${{yAnchor.toFixed(2)}})">gradient-magnitude percentile (0-100)</text>`;

      const classSpecs = [
        {{ cls: "sim", cum_stats: simCumStats, new_stats: simNewStats }},
        {{ cls: "dissim", cum_stats: dissimCumStats, new_stats: dissimNewStats }},
      ];
      for (const spec of classSpecs) {{
        const style = classStyle[spec.cls];
        for (const d of bandPaths(spec.new_stats, mapX, mapY)) {{
          svg += `<path d="${{d}}" fill="${{style.band_fill}}" stroke="none"></path>`;
        }}
      }}
      for (const spec of classSpecs) {{
        const style = classStyle[spec.cls];
        const cumPath = pathFromSeries(spec.cum_stats, mapX, mapY, "mean");
        if (cumPath) {{
          svg += `<path d="${{cumPath}}" fill="none" stroke="${{style.color}}" stroke-width="2.4" stroke-dasharray="7 5" stroke-linecap="round" opacity="0.82"></path>`;
        }}
      }}
      for (const spec of classSpecs) {{
        const style = classStyle[spec.cls];
        const newPath = pathFromSeries(spec.new_stats, mapX, mapY, "mean");
        if (newPath) {{
          svg += `<path d="${{newPath}}" fill="none" stroke="${{style.color}}" stroke-width="4.2" stroke-linecap="round"></path>`;
        }}
        for (const pt of spec.new_stats) {{
          const x = mapX(pt.epoch);
          const y = mapY(pt.mean);
          const r = aggMarkerRadius(pt.count_sum);
          const tip = `${{style.label}} mean new | epoch ${{pt.epoch}} | rank=${{pt.mean.toFixed(2)}} | newly-forgotten=${{Math.round(pt.count_sum)}}`;
          if (spec.cls === "dissim") {{
            svg += `<rect x="${{(x - r).toFixed(2)}}" y="${{(y - r).toFixed(2)}}" width="${{(2 * r).toFixed(2)}}" height="${{(2 * r).toFixed(2)}}" fill="${{style.color}}" stroke="#ffffff" stroke-width="1.1"><title>${{esc(tip)}}</title></rect>`;
          }} else {{
            svg += `<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="${{r.toFixed(2)}}" fill="${{style.color}}" stroke="#ffffff" stroke-width="1.1"><title>${{esc(tip)}}</title></circle>`;
          }}
        }}
        const last = spec.new_stats.length > 0 ? spec.new_stats[spec.new_stats.length - 1] : null;
        if (last) {{
          const lx = mapX(last.epoch) + 8;
          const ly = mapY(last.mean) - 5;
          svg += `<text x="${{lx.toFixed(2)}}" y="${{ly.toFixed(2)}}" font-size="14" fill="${{style.color}}" font-weight="700">${{style.label}} new ${{last.mean.toFixed(1)}}</text>`;
        }}
      }}
      svg += `</svg>`;

      function firstLastText(series) {{
        if (!series || series.length < 2) return "insufficient data";
        const first = series[0].mean;
        const last = series[series.length - 1].mean;
        const delta = last - first;
        const dir = delta < -1.0 ? "down" : (delta > 1.0 ? "up" : "flat");
        return `${{first.toFixed(1)}} -> ${{last.toFixed(1)}} (${{dir}} by ${{Math.abs(delta).toFixed(1)}})`;
      }}

      function nRangeText(series) {{
        if (!series || series.length === 0) return "n/a";
        const ns = series.map((x) => Math.max(0, Math.round(toFiniteNumber(x.n, 0))));
        return `${{Math.min(...ns)}}-${{Math.max(...ns)}}`;
      }}

      function seriesPathMini(series, mapXMini, mapYMini) {{
        return pathFromSeries(series, mapXMini, mapYMini, "value");
      }}

      function firstLastDataset(series) {{
        if (!series || series.length === 0) return "no events";
        const first = series[0].value;
        const last = series[series.length - 1].value;
        return `${{first.toFixed(1)}} -> ${{last.toFixed(1)}}`;
      }}

      function renderMiniDataset(ds) {{
        const w = 520;
        const h = 220;
        const m = {{ top: 12, right: 10, bottom: 26, left: 34 }};
        const pw = w - m.left - m.right;
        const ph = h - m.top - m.bottom;
        const mapXMini = (v) => scaleLinear(v, 1, epochCeil, m.left, m.left + pw);
        const mapYMini = (v) => scaleLinear(v, 0, 100, m.top + ph, m.top);
        const xMid = Math.max(1, Math.round((epochCeil + 1) / 2));
        const miniXTicks = Array.from(new Set([1, xMid, epochCeil])).sort((a, b) => a - b);
        let inner = `<svg class="evo-mini-chart" viewBox="0 0 ${{w}} ${{h}}" role="img" aria-label="${{esc(ds.dataset_label)}} rank evolution">`;
        for (const t of [0, 50, 100]) {{
          const y = mapYMini(t);
          inner += `<line x1="${{m.left}}" y1="${{y.toFixed(2)}}" x2="${{(m.left + pw).toFixed(2)}}" y2="${{y.toFixed(2)}}" stroke="#edf3f8" stroke-width="1"></line>`;
        }}
        for (const t of miniXTicks) {{
          const x = mapXMini(t);
          inner += `<line x1="${{x.toFixed(2)}}" y1="${{m.top}}" x2="${{x.toFixed(2)}}" y2="${{(m.top + ph).toFixed(2)}}" stroke="#f2f6fb" stroke-width="1"></line>`;
          inner += `<text x="${{x.toFixed(2)}}" y="${{(h - 7).toFixed(2)}}" font-size="10.5" fill="#5d7488" text-anchor="middle">${{t}}</text>`;
        }}
        inner += `<line x1="${{m.left}}" y1="${{m.top}}" x2="${{m.left}}" y2="${{(m.top + ph).toFixed(2)}}" stroke="#708ba3" stroke-width="1.2"></line>`;
        inner += `<line x1="${{m.left}}" y1="${{(m.top + ph).toFixed(2)}}" x2="${{(m.left + pw).toFixed(2)}}" y2="${{(m.top + ph).toFixed(2)}}" stroke="#708ba3" stroke-width="1.2"></line>`;

        const simCum = seriesPathMini(ds.sim_cum, mapXMini, mapYMini);
        if (simCum) inner += `<path d="${{simCum}}" fill="none" stroke="${{classStyle.sim.color}}" stroke-width="1.6" stroke-dasharray="6 4" opacity="0.75"></path>`;
        const dissimCum = seriesPathMini(ds.dissim_cum, mapXMini, mapYMini);
        if (dissimCum) inner += `<path d="${{dissimCum}}" fill="none" stroke="${{classStyle.dissim.color}}" stroke-width="1.6" stroke-dasharray="6 4" opacity="0.75"></path>`;
        const simNew = seriesPathMini(ds.sim_new, mapXMini, mapYMini);
        if (simNew) inner += `<path d="${{simNew}}" fill="none" stroke="${{classStyle.sim.color}}" stroke-width="2.2"></path>`;
        const dissimNew = seriesPathMini(ds.dissim_new, mapXMini, mapYMini);
        if (dissimNew) inner += `<path d="${{dissimNew}}" fill="none" stroke="${{classStyle.dissim.color}}" stroke-width="2.2"></path>`;
        inner += `</svg>`;
        const trendTxt = `SIM ${{firstLastDataset(ds.sim_new)}} | DISSIM ${{firstLastDataset(ds.dissim_new)}}`;
        return `<section class="evo-mini-panel"><h4 class="evo-mini-title">${{esc(ds.dataset_label)}}</h4>${{inner}}<p class="evo-mini-note">${{esc(trendTxt)}}</p></section>`;
      }}

      const simTrend = firstLastText(simNewStats);
      const dissimTrend = firstLastText(dissimNewStats);
      const datasetLabels = datasets.map((ds) => ds.dataset_label).filter((x) => x);
      const miniPanels = datasets.map((ds) => renderMiniDataset(ds)).join("");

      let html = `<div class="evo-grad-deck"><div class="evo-grad-main">${{svg}}</div>`;
      html += `<div class="evo-legend">`;
      html += `<span class="evo-legend-item"><span class="evo-line-sample" style="border-top-color:#0E5E5A"></span>SIM mean new-rank</span>`;
      html += `<span class="evo-legend-item"><span class="evo-line-sample" style="border-top-color:#A34D1A"></span>DISSIM mean new-rank</span>`;
      html += `<span class="evo-legend-item"><span class="evo-line-sample dissim" style="border-top-color:#0E5E5A"></span>SIM mean cumulative-rank</span>`;
      html += `<span class="evo-legend-item"><span class="evo-line-sample dissim" style="border-top-color:#A34D1A"></span>DISSIM mean cumulative-rank</span>`;
      html += `<span class="evo-legend-item">Shaded bands = IQR across datasets</span>`;
      html += `</div>`;
      html += `<div class="evo-legend">` + datasets.map((ds) => `<span class="evo-legend-item"><span class="evo-chip" style="background:${{esc(ds.color)}}"></span>${{esc(ds.dataset_label)}}</span>`).join("") + `</div>`;
      html += `<div class="evo-small-grid">${{miniPanels}}</div>`;
      html += `<p class="evo-grad-note"><strong>Rank (simple):</strong> inside each dataset/class, sort questions by <code>|g_i|</code> from largest to smallest. High rank means forgetting is still concentrated on stronger-gradient items.</p>`;
      html += `<p class="evo-grad-note"><strong>Math:</strong> <code>rank_pct(q)=100*(1-(r(q)-1)/(N-1))</code>, where <code>r(q)</code> is position in that sorted list and <code>N</code> is class size.</p>`;
      html += `<p class="evo-grad-note"><strong>Observed trend:</strong> SIM ${{esc(simTrend)}}; DISSIM ${{esc(dissimTrend)}}. Coverage per epoch (datasets): SIM ${{esc(nRangeText(simNewStats))}}, DISSIM ${{esc(nRangeText(dissimNewStats))}}. Datasets: ${{esc(datasetLabels.join(", "))}}.</p>`;
      html += `</div>`;
      return html;
    }}

    function renderTable1Evolution(table) {{
      const groups = collectEvolutionGroups(table);
      if (groups.length === 0) {{
        return `<div class="section evo-wrap"><h2>${{esc(table.title || "Table 1 Evolution")}}</h2><p class="evo-caption">No epoch-wise data available yet.</p></div>`;
      }}
      const epochMax = Math.max(
        1,
        toFiniteNumber(table.max_epoch, 1),
        ...groups.flatMap((group) => (group.points || []).map((point) => toFiniteNumber(point.epoch, 0))),
      );

      const byDataset = new Map();
      for (const group of groups) {{
        if (!byDataset.has(group.dataset)) {{
          byDataset.set(group.dataset, {{
            dataset: group.dataset,
            label: group.dataset_label,
            color: group.color,
          }});
        }}
      }}
      const orderedDatasets = (table.dataset_order || []).length > 0 ? (table.dataset_order || []) : Array.from(byDataset.keys());
      const datasetLegend = [];
      for (const ds of orderedDatasets) {{
        const item = byDataset.get(ds);
        if (item) datasetLegend.push(item);
      }}
      for (const [ds, item] of byDataset.entries()) {{
        if (!orderedDatasets.includes(ds)) datasetLegend.push(item);
      }}

      const datasetLegendHtml = datasetLegend
        .map(
          (item) =>
            `<span class="evo-legend-item"><span class="evo-chip" style="background:${{esc(item.color)}}"></span>${{esc(item.label)}}</span>`,
        )
        .join("");
      const classLegendHtml = `
        <span class="evo-legend-item"><span class="evo-line-sample"></span><span class="evo-marker-sample"></span>SIM (solid + circle)</span>
        <span class="evo-legend-item"><span class="evo-line-sample dissim"></span><span class="evo-marker-sample square"></span>DISSIM (dashed + square)</span>
      `;

      const targetEpoch = toFiniteNumber(table.target_epoch, null);
      const metaParts = [
        `${{datasetLegend.length}} dataset(s)`,
        `max epoch seen: ${{Math.round(epochMax)}}`,
        "x-axis: point index",
        "left-to-right: increasing training iteration",
        "y-axis: forgotten questions (%)",
      ];
      if (targetEpoch !== null) {{
        metaParts.push(`target epoch: ${{Math.round(targetEpoch)}}`);
      }}

      let html = `<div class="section evo-wrap"><h2>${{esc(table.title || "Table 1 Evolution")}}</h2>`;
      if (table.caption) {{
        html += `<p class="evo-caption">${{esc(table.caption)}}</p>`;
      }}
      html += `<div class="evo-panels">`;
      html += `<section class="evo-panel"><h3>Forgetting Dynamics by Point Index</h3>${{renderEvolutionPrimarySvg(groups, epochMax)}}</section>`;
      html += `<section class="evo-panel"><h3>Gradient Similarity Evolution: Rank of Newly Forgotten Samples</h3>${{renderEvolutionGradSvg(groups, epochMax)}}</section>`;
      html += `</div>`;
      html += `<div class="evo-legend">${{datasetLegendHtml}}</div>`;
      html += `<div class="evo-legend">${{classLegendHtml}}</div>`;
      html += `<p class="evo-meta">${{esc(metaParts.join(" | "))}}</p>`;
      html += `</div>`;
      return html;
    }}

    function renderPaperTable(table) {{
      if (table.paper_table === "table1") {{
        return renderPaperTable1(table);
      }}
      if (table.paper_table === "table2") {{
        return renderPaperTable2(table);
      }}
      if (table.paper_table === "table3") {{
        return renderPaperTable3(table);
      }}
      if (table.paper_table === "table4") {{
        return renderPaperTable4(table);
      }}
      return renderTable(table);
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

    function normalizeHeading(line) {{
      let title = String(line || "").trim();
      title = title.replace(/^[0-9]+\\)\\s*/, "");
      title = title.replace(/:$/, "");
      return title;
    }}

    function panelTone(title) {{
      const t = String(title || "").toLowerCase();
      if (t.includes("can safely conclude")) {{
        return "tone-good";
      }}
      if (t.includes("cannot claim")) {{
        return "tone-caution";
      }}
      return "";
    }}

    function parseAppendix(content) {{
      const lines = String(content || "").split(/\\r?\\n/);
      const sections = [];
      const intro = [];
      let current = null;

      for (const raw of lines) {{
        const line = raw.trim();
        if (!line) {{
          continue;
        }}
        if (line.toLowerCase().startsWith("quick read")) {{
          intro.push(line.replace(/:$/, ""));
          continue;
        }}

        const isHeading = (/^[0-9]+\\)\\s+/.test(line) || (line.endsWith(":") && !line.startsWith("- ")));
        if (isHeading) {{
          current = {{ title: normalizeHeading(line), bullets: [], paragraphs: [] }};
          sections.push(current);
          continue;
        }}

        if (line.startsWith("- ")) {{
          if (!current) {{
            current = {{ title: "Summary", bullets: [], paragraphs: [] }};
            sections.push(current);
          }}
          current.bullets.push(line.slice(2).trim());
          continue;
        }}

        if (!current) {{
          intro.push(line);
        }} else {{
          current.paragraphs.push(line);
        }}
      }}

      return {{ intro, sections }};
    }}

    function renderBeautifulAppendix(content) {{
      const parsed = parseAppendix(content);
      let html = `<div class="appendix-shell">`;
      html += `<span class="appendix-kicker">Interpretation Layer</span>`;
      html += `<h2 class="appendix-title">Paper Comparison Appendix</h2>`;
      const subtitle = parsed.intro.length > 0 ? parsed.intro.join(" ") : "Simple explanation of what these results mean.";
      html += `<p class="appendix-subtitle">${{esc(subtitle)}}</p>`;
      html += `<div class="appendix-grid">`;

      for (const section of parsed.sections) {{
        const tone = panelTone(section.title);
        html += `<section class="appendix-panel ${{esc(tone)}}">`;
        html += `<h3>${{esc(section.title)}}</h3>`;
        for (const para of section.paragraphs || []) {{
          html += `<p>${{esc(para)}}</p>`;
        }}
        if ((section.bullets || []).length > 0) {{
          html += `<ul class="appendix-list">`;
          for (const item of section.bullets) {{
            html += `<li>${{esc(item)}}</li>`;
          }}
          html += `</ul>`;
        }}
        html += `</section>`;
      }}

      html += `</div></div>`;
      return html;
    }}

    function renderProject(project) {{
      const statusEl = document.getElementById("projectStatus");
      const status = String(project.status || "unknown").toLowerCase();
      const badgeTone = status === "error" ? "error" : (status === "stale" ? "warn" : "ok");
      statusEl.textContent = status;
      statusEl.className = `badge status-${{badgeTone}}`;

      const errEl = document.getElementById("projectError");
      if (status === "error") {{
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

      const texts = project.texts || [];
      const appendix = texts.find((block) => (block.title || "") === "Paper Comparison Appendix (Simple)");
      const appendixEl = document.getElementById("appendix");
      if (appendix) {{
        appendixEl.innerHTML = renderBeautifulAppendix(appendix.content || "");
      }} else {{
        appendixEl.innerHTML = "";
      }}

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
      document.getElementById("tables").innerHTML = tables
        .map((table) => {{
          if (table.viz === "table1_evolution") {{
            return renderTable1Evolution(table);
          }}
          return table.paper_table ? renderPaperTable(table) : renderTable(table);
        }})
        .join("");

      document.getElementById("texts").innerHTML = texts
        .filter((block) => (block.title || "") !== "Paper Comparison Appendix (Simple)")
        .map((block) => `
        <div class="section">
          <h2>${{esc(block.title || "Text")}}</h2>
          <pre>${{esc(block.content || "")}}</pre>
        </div>
      `)
        .join("");
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
    output_json = Path(args.output_json)

    previous_projects: dict[str, dict[str, Any]] = {}
    if output_json.exists():
        try:
            previous_payload = json.loads(output_json.read_text(encoding="utf-8"))
            for prev in previous_payload.get("projects", []):
                project_id = str(prev.get("project_id", "")).strip()
                if project_id:
                    previous_projects[project_id] = prev
        except Exception:  # noqa: BLE001
            previous_projects = {}

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
            project_id = project.get("id", "unknown")
            now = datetime.now(timezone.utc).isoformat()
            previous = previous_projects.get(str(project_id))
            if previous and str(previous.get("status", "")).lower() in {"ok", "stale"}:
                stale = json.loads(json.dumps(previous))
                stale["project_id"] = project_id
                stale["project_name"] = project.get("name", project_id)
                stale["collector"] = project.get("collector", "")
                stale["status"] = "stale"
                stale["error"] = ""
                stale["stale"] = True
                stale["stale_checked_at_utc"] = now
                stale["fallback_reason"] = str(exc)
                last_ok = stale.get("collected_at_utc", "unknown")

                cards = [c for c in stale.get("cards", []) if str(c.get("label", "")) != "Collection Warning"]
                cards.insert(
                    0,
                    {
                        "label": "Collection Warning",
                        "value": (
                            f"Live probe failed at {now}. "
                            f"Showing last successful snapshot from {last_ok}.\n"
                            f"Reason: {exc}"
                        ),
                    },
                )
                stale["cards"] = cards

                texts = [t for t in stale.get("texts", []) if str(t.get("title", "")) != "Collection Warning"]
                texts.insert(
                    0,
                    {
                        "title": "Collection Warning",
                        "content": (
                            f"Live probe failed at {now}.\n"
                            f"Showing cached data from {last_ok}.\n\n"
                            f"Reason:\n{exc}"
                        ),
                    },
                )
                stale["texts"] = texts
                stale["links"] = [x for x in stale.get("links", []) if is_link_visible(x)]
                out_projects.append(stale)
                continue

            out_projects.append(
                {
                    "project_id": project_id,
                    "project_name": project.get("name", project_id),
                    "collector": project.get("collector", ""),
                    "status": "error",
                    "error": str(exc),
                    "collected_at_utc": now,
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
