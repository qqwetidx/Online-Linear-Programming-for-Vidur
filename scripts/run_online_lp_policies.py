import argparse
import csv
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ONLINE_LP_POLICIES = ["dual_ahd"]
BASELINE_SCHEDULERS = ["random", "round_robin", "lor"]


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    idx = int(q * (len(values) - 1))
    return values[idx]


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_request_metrics(path, time_limit):
    total_requests = 0
    completed = 0
    tokens_completed = 0
    e2e = []
    sched = []
    prefill = []
    execution = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_requests += 1
            e2e_val = parse_float(row.get("request_e2e_time"))
            completed_flag = parse_float(row.get("request_completed"))
            if completed_flag is not None:
                is_completed = completed_flag > 0
            else:
                is_completed = e2e_val is not None

            if is_completed:
                completed += 1
                tokens_val = parse_float(row.get("request_num_tokens"))
                if tokens_val is not None:
                    tokens_completed += int(tokens_val)

            if e2e_val is None:
                continue
            e2e.append(e2e_val)

            sched_val = parse_float(row.get("request_scheduling_delay"))
            if sched_val is not None:
                sched.append(sched_val)
            prefill_val = parse_float(row.get("prefill_e2e_time"))
            if prefill_val is not None:
                prefill.append(prefill_val)
            execution_val = parse_float(row.get("request_execution_time"))
            if execution_val is not None:
                execution.append(execution_val)

    return {
        "total_requests": total_requests,
        "completed_requests": completed,
        "completion_rate": completed / total_requests if total_requests else 0.0,
        "tokens_completed": tokens_completed,
        "token_throughput_tps": tokens_completed / time_limit if time_limit else 0.0,
        "token_throughput_tpm": tokens_completed * (60.0 / time_limit)
        if time_limit
        else 0.0,
        "qpm": completed * (60.0 / time_limit) if time_limit else 0.0,
        "mean_e2e": sum(e2e) / len(e2e) if e2e else None,
        "p95_e2e": quantile(e2e, 0.95),
        "p99_e2e": quantile(e2e, 0.99),
        "mean_scheduling_delay": sum(sched) / len(sched) if sched else None,
        "p95_scheduling_delay": quantile(sched, 0.95),
        "p99_scheduling_delay": quantile(sched, 0.99),
        "mean_prefill_e2e": sum(prefill) / len(prefill) if prefill else None,
        "p95_prefill_e2e": quantile(prefill, 0.95),
        "p99_prefill_e2e": quantile(prefill, 0.99),
        "mean_execution_time": sum(execution) / len(execution) if execution else None,
        "p95_execution_time": quantile(execution, 0.95),
        "p99_execution_time": quantile(execution, 0.99),
    }


def count_csv_rows(path: Path) -> int:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return 0
        return sum(1 for _ in reader)


def load_latency_series(path):
    latencies = []
    if not path or not path.exists():
        return latencies
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            e2e_val = parse_float(row.get("request_e2e_time"))
            if e2e_val is not None:
                latencies.append(e2e_val)
    return latencies


def plot_summary_metrics(summary_path, output_dir):
    rows = list(csv.DictReader(summary_path.open(newline="")))
    if not rows:
        return

    metric_keys = [
        ("Completed Requests", "completed_requests"),
        ("Token Throughput (tps)", "token_throughput_tps"),
        ("QPM", "qpm"),
        ("Mean E2E (s)", "mean_e2e"),
        ("P95 E2E (s)", "p95_e2e"),
        ("P99 E2E (s)", "p99_e2e"),
        ("Mean Scheduling Delay (s)", "mean_scheduling_delay"),
        ("P95 Scheduling Delay (s)", "p95_scheduling_delay"),
        ("P99 Scheduling Delay (s)", "p99_scheduling_delay"),
        ("Mean Prefill E2E (s)", "mean_prefill_e2e"),
        ("P95 Prefill E2E (s)", "p95_prefill_e2e"),
        ("P99 Prefill E2E (s)", "p99_prefill_e2e"),
        ("Mean Execution Time (s)", "mean_execution_time"),
        ("P95 Execution Time (s)", "p95_execution_time"),
        ("P99 Execution Time (s)", "p99_execution_time"),
    ]

    algos = [r["algorithm"] for r in rows]
    colors = ["#c0c0c0" for _ in algos]
    for i, name in enumerate(algos):
        if "dual_ahd" in name:
            colors[i] = "#e74c3c"

    cols = 3
    rows_count = int(math.ceil(len(metric_keys) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(16, 4.2 * rows_count))
    axes = np.array(axes).reshape(-1)

    for ax, (title, key) in zip(axes, metric_keys):
        values = []
        for row in rows:
            val = parse_float(row.get(key))
            values.append(val if val is not None else float("nan"))
        ax.bar(range(len(algos)), values, color=colors)
        ax.set_title(title)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, rotation=30, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for ax in axes[len(metric_keys) :]:
        ax.axis("off")

    fig.suptitle(
        "Online LP Summary (Latency Includes Incomplete Requests)", fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_dir / "summary_plot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_latency_distributions(algorithm_rows, output_dir):
    entries = []
    for row in algorithm_rows:
        metrics_path = row.get("metrics_path")
        if metrics_path is not None and not isinstance(metrics_path, Path):
            metrics_path = Path(metrics_path)
        if metrics_path is None or not metrics_path.exists():
            base_dir = row.get("base_dir")
            if base_dir is not None and not isinstance(base_dir, Path):
                base_dir = Path(base_dir)
            run_dir = find_run_dir(base_dir) if base_dir else None
            metrics_path = run_dir / "request_metrics.csv" if run_dir else None
        latencies = load_latency_series(metrics_path)
        if latencies:
            entries.append((row["label"], latencies))

    if not entries:
        return

    fig_cdf, ax_cdf = plt.subplots(figsize=(8, 5))
    for label, values in entries:
        vals = np.array(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        color = "#e74c3c" if "dual_ahd" in label else None
        line_width = 2.4 if color else 1.6
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax_cdf.plot(sorted_vals, cdf, label=label, color=color, linewidth=line_width)

    ax_cdf.set_title("E2E Latency CDF (All Strategies)")
    ax_cdf.set_xlabel("Latency (s)")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.grid(True, linestyle="--", alpha=0.4)
    ax_cdf.legend(fontsize=9)
    fig_cdf.tight_layout()
    fig_cdf.savefig(output_dir / "latency_cdf.png", dpi=150)
    plt.close(fig_cdf)

    cols = 2 if len(entries) > 1 else 1
    rows_count = int(math.ceil(len(entries) / cols))
    fig_hist, axes_hist = plt.subplots(
        rows_count, cols, figsize=(12, 4.2 * rows_count)
    )
    axes_hist = np.array(axes_hist).reshape(-1)

    for idx, (label, values) in enumerate(entries):
        ax_hist = axes_hist[idx]
        vals = np.array(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax_hist.axis("off")
            continue

        color = "#e74c3c" if "dual_ahd" in label else "#7f8c8d"
        ax_hist.hist(vals, bins=50, color=color, alpha=0.85)
        ax_hist.set_title(f"{label} - E2E Histogram")
        ax_hist.set_xlabel("Latency (s)")
        ax_hist.set_ylabel("Count")
        ax_hist.grid(True, linestyle="--", alpha=0.4)

    for ax in axes_hist[len(entries) :]:
        ax.axis("off")

    fig_hist.tight_layout()
    fig_hist.savefig(output_dir / "latency_hist.png", dpi=150)
    plt.close(fig_hist)


def run_simulation(args, output_dir, scheduler_type, policy=None):
    cmd = [
        sys.executable,
        "-m",
        "vidur.main",
        "--time_limit",
        str(args.time_limit),
        "--cluster_config_num_replicas",
        str(args.num_replicas),
        "--poisson_request_interval_generator_config_qps",
        str(args.qps),
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_store_plots",
        "--metrics_config_output_dir",
        str(output_dir),
    ]
    if args.replica_device:
        cmd.extend(["--replica_config_device", args.replica_device])
    if args.memory_margin_fraction is not None:
        cmd.extend(
            [
                "--replica_config_memory_margin_fraction",
                str(args.memory_margin_fraction),
            ]
        )
    scheduler_prefixes = [
        "sarathi_scheduler_config",
        "vllm_scheduler_config",
        "lightllm_scheduler_config",
        "orca_scheduler_config",
        "faster_transformer_scheduler_config",
    ]
    if args.batch_size_cap is not None:
        for prefix in scheduler_prefixes:
            cmd.extend([f"--{prefix}_batch_size_cap", str(args.batch_size_cap)])
    if args.num_blocks is not None:
        for prefix in scheduler_prefixes:
            cmd.extend([f"--{prefix}_num_blocks", str(args.num_blocks)])
    if args.block_size is not None:
        for prefix in scheduler_prefixes:
            cmd.extend([f"--{prefix}_block_size", str(args.block_size)])
    if args.length_csv:
        cmd.extend(
            [
                "--synthetic_request_generator_config_num_requests",
                str(args.num_requests),
                "--length_generator_config_type",
                "trace",
                "--trace_request_length_generator_config_trace_file",
                str(args.length_csv),
                "--trace_request_length_generator_config_prefill_column",
                args.length_csv_prefill_column,
                "--trace_request_length_generator_config_decode_column",
                args.length_csv_decode_column,
            ]
        )
        if args.length_csv_shuffle:
            cmd.append("--trace_request_length_generator_config_shuffle")
        else:
            cmd.append("--no-trace_request_length_generator_config_shuffle")
    else:
        cmd.extend(
            [
                "--synthetic_request_generator_config_duration",
                str(args.duration),
                "--synthetic_request_generator_config_num_requests",
                str(args.num_requests),
            ]
        )
    if scheduler_type == "online_lp":
        cmd.extend(
            [
                "--global_scheduler_config_type",
                "online_lp",
                "--online_lp_global_scheduler_config_policy",
                policy,
                "--online_lp_global_scheduler_config_horizon",
                str(args.horizon),
                "--online_lp_global_scheduler_config_time_step_seconds",
                str(args.time_step_seconds),
                "--online_lp_global_scheduler_config_total_jobs",
                str(args.total_jobs),
                "--online_lp_global_scheduler_config_grad_steps",
                str(args.grad_steps),
                "--online_lp_global_scheduler_config_step_size",
                str(args.step_size),
                "--online_lp_global_scheduler_config_batch_size",
                str(args.batch_size),
                "--online_lp_global_scheduler_config_mem_scale_factor",
                str(args.mem_scale_factor),
                "--online_lp_global_scheduler_config_backlog_gain",
                str(args.backlog_gain),
                "--online_lp_global_scheduler_config_backlog_slack",
                str(args.backlog_slack),
                "--online_lp_global_scheduler_config_length_penalty",
                str(args.length_penalty),
                "--online_lp_global_scheduler_config_reward_normalizer",
                str(args.reward_normalizer),
                "--online_lp_global_scheduler_config_alpha",
                str(args.alpha),
                "--online_lp_global_scheduler_config_beta",
                str(args.beta),
                "--online_lp_global_scheduler_config_gamma",
                str(args.gamma),
                "--online_lp_global_scheduler_config_sigma",
                str(args.sigma),
                "--online_lp_global_scheduler_config_goodput_window",
                str(args.goodput_window),
                "--online_lp_global_scheduler_config_goodput_weight",
                str(args.goodput_weight),
                "--online_lp_global_scheduler_config_max_waiting",
                str(args.max_waiting),
            ]
        )
    else:
        cmd.extend(["--global_scheduler_config_type", scheduler_type])

    if args.no_cache:
        cmd.append("--random_forrest_execution_time_predictor_config_no_cache")

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")

    result = subprocess.run(cmd, cwd=args.repo_root, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Simulation failed: {' '.join(cmd)}")


def find_run_dir(base_dir):
    if not base_dir.exists():
        return None
    subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    subdirs = sorted(subdirs, key=lambda p: p.name)
    for candidate in reversed(subdirs):
        if (candidate / "request_metrics.csv").exists():
            return candidate
    return subdirs[-1]


def write_summary(path, rows):
    if not rows:
        return
    preferred = [
        "algorithm",
        "run_dir",
        "total_requests",
        "completed_requests",
        "completion_rate",
        "tokens_completed",
        "token_throughput_tps",
        "token_throughput_tpm",
        "qpm",
        "mean_e2e",
        "p95_e2e",
        "p99_e2e",
        "mean_scheduling_delay",
        "p95_scheduling_delay",
        "p99_scheduling_delay",
        "mean_prefill_e2e",
        "p95_prefill_e2e",
        "p99_prefill_e2e",
        "mean_execution_time",
        "p95_execution_time",
        "p99_execution_time",
        "error",
    ]
    fieldnames = [key for key in preferred if any(key in row for row in rows)]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Run Online LP policies (and optional baselines) and compare results."
    )
    parser.add_argument("--time-limit", type=int, default=180)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--num-replicas", type=int, default=4)
    parser.add_argument(
        "--replica-device",
        type=str,
        default="",
        help="Override replica device type (e.g., a40, a100, h100).",
    )
    parser.add_argument("--qps", type=float, default=20.0)
    parser.add_argument("--time-step-seconds", type=float, default=0.03)
    parser.add_argument("--horizon", type=int, default=30000)
    parser.add_argument("--total-jobs", type=int, default=1000)
    parser.add_argument("--grad-steps", type=int, default=5)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mem-scale-factor", type=float, default=1.0)
    parser.add_argument("--backlog-gain", type=float, default=1.0)
    parser.add_argument("--backlog-slack", type=float, default=1.0)
    parser.add_argument("--length-penalty", type=float, default=0.0)
    parser.add_argument("--reward-normalizer", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--goodput-window", type=int, default=0)
    parser.add_argument("--goodput-weight", type=float, default=0.0)
    parser.add_argument("--max-waiting", type=int, default=0)
    parser.add_argument(
        "--length-csv",
        type=str,
        default="",
        help="CSV file with per-request length columns (enables trace length generator).",
    )
    parser.add_argument(
        "--length-csv-prefill-column",
        type=str,
        default="num_prefill_tokens",
        help="Prefill column name in length CSV.",
    )
    parser.add_argument(
        "--length-csv-decode-column",
        type=str,
        default="num_decode_tokens",
        help="Decode column name in length CSV.",
    )
    parser.add_argument(
        "--length-csv-shuffle",
        action="store_true",
        help="Shuffle the length CSV instead of using row order.",
    )
    parser.add_argument(
        "--batch-size-cap",
        type=int,
        default=None,
        help="Cap replica scheduler batch size.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Override total KV cache blocks for replica scheduler.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Override KV cache block size (tokens per block).",
    )
    parser.add_argument(
        "--memory-margin-fraction",
        type=float,
        default=None,
        help="Fraction of GPU memory reserved (higher = less usable memory).",
    )
    parser.add_argument(
        "--repro-input-output-csv",
        action="store_true",
        help="Use input_output_word_counts10000+(1).csv with input/output columns to reproduce results.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--repo-root", type=str, default=str(Path(__file__).resolve().parents[1])
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="simulator_output/repro_window30_r4_qps80_online_lp",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    if not repo_root.exists():
        raise FileNotFoundError(f"Repo root not found: {repo_root}")

    if args.repro_input_output_csv:
        args.length_csv = "input_output_word_counts10000+(1).csv"
        args.length_csv_prefill_column = "input"
        args.length_csv_decode_column = "output"
        args.length_csv_shuffle = False
        args.num_requests = 1254
        args.force = True

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.length_csv:
        length_csv_path = Path(args.length_csv)
        if not length_csv_path.is_absolute():
            length_csv_path = repo_root / length_csv_path
        if not length_csv_path.exists():
            raise FileNotFoundError(f"Length CSV not found: {length_csv_path}")
        length_rows = count_csv_rows(length_csv_path)
        if length_rows <= 0:
            raise ValueError(f"Length CSV is empty: {length_csv_path}")
        if args.num_requests > length_rows:
            print(
                f"Length CSV has {length_rows} rows; truncating num_requests from "
                f"{args.num_requests} to {length_rows}."
            )
        args.num_requests = min(args.num_requests, length_rows)
        args.length_csv = str(length_csv_path)

    if args.horizon <= 0:
        args.horizon = max(1, int(math.ceil(args.time_limit / args.time_step_seconds)))
    if args.total_jobs <= 0:
        if args.length_csv:
            args.total_jobs = max(1, int(args.num_requests))
        else:
            args.total_jobs = max(1, int(args.qps * args.duration))

    algorithms = []
    if not args.no_baselines:
        for scheduler in BASELINE_SCHEDULERS:
            algorithms.append(
                {
                    "label": f"baseline_{scheduler}",
                    "scheduler": scheduler,
                    "policy": None,
                }
            )
    for policy in ONLINE_LP_POLICIES:
        algorithms.append(
            {
                "label": f"online_lp_{policy}",
                "scheduler": "online_lp",
                "policy": policy,
            }
        )

    run_errors = {}
    jobs = []
    for item in algorithms:
        base_dir = output_root / item["label"]
        run_dir = find_run_dir(base_dir)
        metrics_path = run_dir / "request_metrics.csv" if run_dir else None
        item["base_dir"] = base_dir
        item["run_dir"] = run_dir
        item["metrics_path"] = metrics_path
        item["needs_run"] = (
            not args.summary_only
            and (not metrics_path or not metrics_path.exists() or args.force)
        )
        if item["needs_run"]:
            jobs.append(item)

    if jobs:
        max_workers = max(1, min(args.max_workers, len(jobs)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    run_simulation,
                    args,
                    job["base_dir"],
                    job["scheduler"],
                    job["policy"],
                ): job
                for job in jobs
            }
            for future in as_completed(future_map):
                job = future_map[future]
                try:
                    future.result()
                except Exception as exc:
                    run_errors[job["label"]] = str(exc)

    summary_rows = []
    for item in algorithms:
        run_dir = find_run_dir(item["base_dir"])
        metrics_path = run_dir / "request_metrics.csv" if run_dir else None
        row = {
            "algorithm": item["label"],
            "run_dir": str(run_dir) if run_dir else "",
        }
        if metrics_path and metrics_path.exists():
            row.update(parse_request_metrics(metrics_path, args.time_limit))
        else:
            row.update({"error": "missing request_metrics.csv"})
        if item["label"] in run_errors:
            row["error"] = run_errors[item["label"]]
        summary_rows.append(row)

    summary_path = output_root / "summary.csv"
    write_summary(summary_path, summary_rows)
    print(f"Summary written to: {summary_path}")
    for row in summary_rows:
        algo = row.get("algorithm")
        completed = row.get("completed_requests")
        total = row.get("total_requests")
        token_tps = row.get("token_throughput_tps")
        qpm = row.get("qpm")
        print(f"{algo}: completed={completed}/{total}, token_tps={token_tps}, qpm={qpm}")

    plot_summary_metrics(summary_path, output_root)
    plot_latency_distributions(algorithms, output_root)


if __name__ == "__main__":
    main()
