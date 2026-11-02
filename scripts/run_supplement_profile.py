"""Sample a local command's process-tree memory without changing the workload."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    peak_rss = peak_private = peak_count = samples = 0
    min_available = psutil.virtual_memory().available
    names = set()
    with (args.output_dir / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=args.cwd, stdout=log,
                                   stderr=subprocess.STDOUT)
        root = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                members = [root] + root.children(recursive=True)
            except psutil.NoSuchProcess:
                break
            rss = private = count = 0
            for member in members:
                try:
                    info = member.memory_info()
                    rss += info.rss
                    private += getattr(info, "private", 0)
                    names.add(member.name())
                    count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            peak_rss = max(peak_rss, rss)
            peak_private = max(peak_private, private)
            peak_count = max(peak_count, count)
            min_available = min(min_available, psutil.virtual_memory().available)
            samples += 1
            time.sleep(0.25)
        exit_code = process.wait()
    result = {
        "command": command, "cwd": str(args.cwd), "exit_code": exit_code,
        "wall_seconds": time.perf_counter() - start,
        "peak_sampled_tree_rss_bytes": peak_rss,
        "peak_sampled_tree_private_bytes": peak_private,
        "minimum_host_available_bytes": min_available,
        "peak_process_count": peak_count, "samples": samples,
        "observed_process_names": sorted(names),
        "scope": "250ms sampled process-tree sums; RSS may double-count shared pages; not an exact OS high-water mark",
    }
    (args.output_dir / "resources.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
