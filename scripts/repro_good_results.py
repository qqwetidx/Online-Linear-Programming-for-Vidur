import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path


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


def run_simulation(args, output_dir, extra_args):
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
        "--synthetic_request_generator_config_duration",
        str(args.duration),
        "--synthetic_request_generator_config_num_requests",
        str(args.num_requests),
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_store_plots",
        "--metrics_config_output_dir",
        str(output_dir),
    ]
    cmd.extend(extra_args)

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
        description="Reproduce the 30s, QPS=80 good-results setup for all schedulers."
    )
    parser.add_argument("--time-limit", type=int, default=30)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--num-replicas", type=int, default=4)
    parser.add_argument("--qps", type=float, default=80.0)
    parser.add_argument("--time-step-seconds", type=float, default=0.05)
    parser.add_argument("--online-lp-horizon", type=int, default=600)
    parser.add_argument("--online-lp-total-jobs", type=int, default=10000)
    parser.add_argument("--online-lp-grad-steps", type=int, default=5)
    parser.add_argument("--online-lp-step-size", type=float, default=0.01)
    parser.add_argument("--online-lp-batch-size", type=int, default=64)
    parser.add_argument("--online-lp-mem-scale-factor", type=float, default=1.0)
    parser.add_argument("--online-lp-backlog-gain", type=float, default=1.0)
    parser.add_argument("--online-lp-backlog-slack", type=float, default=1.0)
    parser.add_argument("--online-lp-length-penalty", type=float, default=0.0)
    parser.add_argument("--online-lp-reward-normalizer", type=float, default=1.0)
    parser.add_argument("--online-lp-max-waiting", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--repo-root", type=str, default=str(Path(__file__).resolve().parents[1])
    )
    parser.add_argument("--output-root", type=str, default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    if not repo_root.exists():
        raise FileNotFoundError(f"Repo root not found: {repo_root}")

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = repo_root / "simulator_output" / "window30_r4_qps80_repro"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.online_lp_horizon <= 0:
        horizon = max(1, int(math.ceil(args.time_limit / args.time_step_seconds)))
    else:
        horizon = args.online_lp_horizon

    if args.online_lp_total_jobs <= 0:
        total_jobs = max(1, int(args.qps * args.duration))
    else:
        total_jobs = args.online_lp_total_jobs

    summary_rows = []

    for scheduler in BASELINE_SCHEDULERS:
        base_dir = output_root / f"baseline_{scheduler}"
        run_dir = find_run_dir(base_dir)
        metrics_path = run_dir / "request_metrics.csv" if run_dir else None
        if (
            not args.summary_only
            and (not metrics_path or not metrics_path.exists() or args.force)
        ):
            run_simulation(
                args,
                base_dir,
                ["--global_scheduler_config_type", scheduler],
            )
            run_dir = find_run_dir(base_dir)
            metrics_path = run_dir / "request_metrics.csv" if run_dir else None
        row = {
            "algorithm": scheduler,
            "run_dir": str(run_dir) if run_dir else "",
        }
        if metrics_path and metrics_path.exists():
            row.update(parse_request_metrics(metrics_path, args.time_limit))
        else:
            row.update({"error": "missing request_metrics.csv"})
        summary_rows.append(row)

    for policy in ONLINE_LP_POLICIES:
        base_dir = output_root / f"online_lp_{policy}"
        run_dir = find_run_dir(base_dir)
        metrics_path = run_dir / "request_metrics.csv" if run_dir else None
        if (
            not args.summary_only
            and (not metrics_path or not metrics_path.exists() or args.force)
        ):
            run_simulation(
                args,
                base_dir,
                [
                    "--global_scheduler_config_type",
                    "online_lp",
                    "--online_lp_global_scheduler_config_policy",
                    policy,
                    "--online_lp_global_scheduler_config_horizon",
                    str(horizon),
                    "--online_lp_global_scheduler_config_time_step_seconds",
                    str(args.time_step_seconds),
                    "--online_lp_global_scheduler_config_total_jobs",
                    str(total_jobs),
                    "--online_lp_global_scheduler_config_grad_steps",
                    str(args.online_lp_grad_steps),
                    "--online_lp_global_scheduler_config_step_size",
                    str(args.online_lp_step_size),
                    "--online_lp_global_scheduler_config_batch_size",
                    str(args.online_lp_batch_size),
                    "--online_lp_global_scheduler_config_mem_scale_factor",
                    str(args.online_lp_mem_scale_factor),
                    "--online_lp_global_scheduler_config_backlog_gain",
                    str(args.online_lp_backlog_gain),
                    "--online_lp_global_scheduler_config_backlog_slack",
                    str(args.online_lp_backlog_slack),
                    "--online_lp_global_scheduler_config_length_penalty",
                    str(args.online_lp_length_penalty),
                    "--online_lp_global_scheduler_config_reward_normalizer",
                    str(args.online_lp_reward_normalizer),
                    "--online_lp_global_scheduler_config_max_waiting",
                    str(args.online_lp_max_waiting),
                ],
            )
            run_dir = find_run_dir(base_dir)
            metrics_path = run_dir / "request_metrics.csv" if run_dir else None
        row = {
            "algorithm": f"online_lp_{policy}",
            "run_dir": str(run_dir) if run_dir else "",
        }
        if metrics_path and metrics_path.exists():
            row.update(parse_request_metrics(metrics_path, args.time_limit))
        else:
            row.update({"error": "missing request_metrics.csv"})
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


if __name__ == "__main__":
    main()
