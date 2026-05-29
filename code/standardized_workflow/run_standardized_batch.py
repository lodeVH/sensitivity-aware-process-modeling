from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROCESS_SCRIPT = REPOSITORY_ROOT / "code" / "standardized_workflow" / "run_standardized_process.py"
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "results" / (
    "standardized_results_windows" if os.name == "nt" else "standardized_results_linux"
)


def parse_item(text: str) -> tuple[str, str]:
    if ":" in text:
        system, regime = text.split(":", 1)
    elif "," in text:
        system, regime = text.split(",", 1)
    else:
        parts = text.split()
        if len(parts) != 2:
            raise ValueError(f"Batch item must look like system:regime, got: {text}")
        system, regime = parts
    return system.strip(), regime.strip()


def write_status(batch_dir: Path, payload: dict) -> None:
    lines = [
        f"# Standardized Batch {payload['batch_id']}",
        "",
        f"- status: {payload['status']}",
        f"- started_at: {payload['started_at']}",
        f"- updated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- jobs_parallel: {payload['jobs_parallel']}",
        f"- current_item: {payload.get('current_item', '')}",
        f"- completed_items: {payload['completed_items']}",
        f"- failed_items: {payload['failed_items']}",
        f"- remaining_items: {payload['remaining_items']}",
        "",
        "## Queue",
    ]
    for item in payload["queue"]:
        marker = item.get("status", "pending")
        lines.append(f"- {marker}: {item['system']} {item['regime']} -> {item['run_id']}")
    lines.append("")
    lines.append("## Results Root")
    lines.append(str(payload["results_root"]))
    lines.append("")
    (batch_dir / "status.md").write_text("\n".join(lines), encoding="utf-8")
    (batch_dir / "batch_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_item(args, batch_dir: Path, item: dict) -> dict:
    stdout_path = batch_dir / f"{item['index']:02d}_{item['system']}_{item['regime']}_stdout.log"
    stderr_path = batch_dir / f"{item['index']:02d}_{item['system']}_{item['regime']}_stderr.log"
    command = [
        args.python,
        str(PROCESS_SCRIPT),
        item["system"],
        item["regime"],
        "--jobs",
        str(args.jobs),
        "--run-id",
        item["run_id"],
        "--results-root",
        str(args.results_root),
    ]
    if args.smoke:
        command.append("--smoke")
    if args.max_files:
        command.extend(["--max-files", str(args.max_files)])

    item["command"] = " ".join(command)
    item["stdout_log"] = str(stdout_path)
    item["stderr_log"] = str(stderr_path)
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, text=True)
    item["runtime_seconds"] = time.perf_counter() - start
    item["returncode"] = completed.returncode
    item["status"] = "completed" if completed.returncode == 0 else "failed"
    return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential batch runner for standardized gradient processes.")
    parser.add_argument("items", nargs="+", help="Items as system:regime, for example lumped:return_to_baseline low_dim:var.")
    parser.add_argument("--jobs", type=int, default=8, help="Parallel workers passed to each standardized process.")
    parser.add_argument("--batch-id", default=datetime.now().strftime("standardized_%Y%m%d_%H%M%S"))
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-files", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_root = args.results_root.expanduser().resolve()
    batch_dir = args.results_root / "_batch_runs" / args.batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    queue = []
    for index, text in enumerate(args.items, start=1):
        system, regime = parse_item(text)
        queue.append(
            {
                "index": index,
                "system": system,
                "regime": regime,
                "run_id": f"{args.batch_id}_{index:02d}_{system}_{regime}",
                "status": "pending",
            }
        )

    payload = {
        "batch_id": args.batch_id,
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "jobs_parallel": args.jobs,
        "results_root": str(args.results_root),
        "batch_dir": str(batch_dir),
        "queue": queue,
        "current_item": "",
        "completed_items": 0,
        "failed_items": 0,
        "remaining_items": len(queue),
    }
    write_status(batch_dir, payload)

    exit_code = 0
    for item in queue:
        item["status"] = "running"
        payload["current_item"] = f"{item['system']} {item['regime']}"
        write_status(batch_dir, payload)
        run_item(args, batch_dir, item)
        if item["status"] == "completed":
            payload["completed_items"] += 1
        else:
            payload["failed_items"] += 1
            exit_code = 1
            if not args.continue_on_failure:
                payload["status"] = "stopped_after_failure"
                payload["remaining_items"] = sum(1 for queued in queue if queued["status"] == "pending")
                write_status(batch_dir, payload)
                (batch_dir / "STANDARDIZED_BATCH_STOPPED.md").write_text(
                    f"Batch stopped after failed item: {item['system']} {item['regime']}\n",
                    encoding="utf-8",
                )
                return exit_code
        payload["remaining_items"] = sum(1 for queued in queue if queued["status"] == "pending")
        write_status(batch_dir, payload)

    payload["status"] = "completed" if exit_code == 0 else "completed_with_failures"
    payload["current_item"] = ""
    payload["remaining_items"] = 0
    write_status(batch_dir, payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
