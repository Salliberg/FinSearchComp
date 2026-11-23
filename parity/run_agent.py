"""Answer FinSearchComp questions with a CLI agent (Claude Code), for Harbor parity.

Upstream answers each question with one model call plus a gateway-provided search
tool, which cannot be run outside that gateway. For parity, both sides run the same
CLI agent on the same instruction, and only the harness differs. This script mirrors
the Harbor task and the Harbor claude-code agent:

- task selection: T2/T3 from data/finsearchcomp_data.json, T1 from
  data/finsearchcomp_akshare_version.json, without (T1)..._096 (指数);
- instruction: parity/instruction.md, the same text as the Harbor task;
- container: python:3.12-slim + Claude Code (parity/Dockerfile), cwd /app,
  1 CPU, 2 GB memory, public network, 3600 s timeout;
- launch: the instruction on stdin to
  `claude --verbose --output-format=stream-json --permission-mode=bypassPermissions --print`
  with IS_SANDBOX=1, CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 and ANTHROPIC_MODEL set.

The answer is the content of /app/answer.md (empty if the agent wrote none). Output
is in chat.py's result format, so finsearchcomp/eval/eval.py grades it unchanged.

Example:
    docker build -t finsearchcomp-parity-claude parity
    export ANTHROPIC_API_KEY=...
    python parity/run_agent.py --model claude-haiku-5-5 --tiers t2,t3 --out parity/runs/run1 --workers 6
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
MAIN_FILE = "finsearchcomp_data.json"
AKSHARE_FILE = "finsearchcomp_akshare_version.json"
INSTRUCTION = (Path(__file__).resolve().parent / "instruction.md").read_text(encoding="utf-8")

# Same exclusion and region rule as the Harbor adapter.
EXCLUDED_T1 = {("(T1)Time_Sensitive_Data_Fetching_096", "指数")}
GLOBAL_T1_TAGS = {"Stock", "Index"}
DEFAULT_PASS_ENV = "ANTHROPIC_API_KEY,ANTHROPIC_BASE_URL"


def sanitize(identifier: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", identifier.lower()).strip("-")


def task_id(record: dict) -> str:
    """Same id as the Harbor task name (without the org prefix)."""
    pid = record["prompt_id"]
    if pid.startswith("(T1)"):
        region = "global" if record.get("tags") in GLOBAL_T1_TAGS else "greater-china"
        return f"{sanitize(pid)}-{region}"
    return sanitize(pid)


def load_tasks(tiers: set[str]) -> list[dict]:
    records = []
    if tiers & {"t2", "t3"}:
        prefixes = tuple(f"(T{t[1]})" for t in sorted(tiers) if t != "t1")
        main = json.loads((DATA_DIR / MAIN_FILE).read_text(encoding="utf-8"))
        records += [r for r in main if r["prompt_id"].startswith(prefixes)]
    if "t1" in tiers:
        ak = json.loads((DATA_DIR / AKSHARE_FILE).read_text(encoding="utf-8"))
        records += [
            r for r in ak
            if r["prompt_id"].startswith("(T1)")
            and (r["prompt_id"], r.get("tags")) not in EXCLUDED_T1
        ]
    return records


def run_one(record: dict, args, out: Path, pass_env: list[str]) -> dict:
    tid = task_id(record)
    work = out / "work" / tid
    app_dir, log_dir = work / "app", work / "logs"
    app_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm", "-i",
        "--name", f"fsc-parity-{out.name}-{tid}"[:120],
        "--cpus", "1", "--memory", "2g",
        "-v", f"{app_dir}:/app", "-v", f"{log_dir}:/logs/agent", "-w", "/app",
        "-e", "IS_SANDBOX=1",
        "-e", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",
        "-e", "CLAUDE_CONFIG_DIR=/logs/agent/sessions",
        "-e", f"ANTHROPIC_MODEL={args.model}",
    ]
    for name in pass_env:
        if os.environ.get(name):
            cmd += ["-e", name]  # value is read from this process's environment
    cmd += [
        args.image, "bash", "-c",
        "mkdir -p $CLAUDE_CONFIG_DIR && "
        "claude --verbose --output-format=stream-json "
        "--permission-mode=bypassPermissions --print 2>&1 | tee /logs/agent/claude-code.txt",
    ]

    instruction = INSTRUCTION.replace("{prompt}", record["prompt"])
    start = time.time()
    status = "ok"
    try:
        proc = subprocess.run(
            cmd, input=instruction, text=True, capture_output=True, timeout=args.timeout
        )
        if proc.returncode != 0:
            status = f"exit {proc.returncode}"
    except subprocess.TimeoutExpired:
        status = "timeout"
        subprocess.run(["docker", "rm", "-f", cmd[cmd.index("--name") + 1]], capture_output=True)

    answer_file = app_dir / "answer.md"
    answer = answer_file.read_text(encoding="utf-8") if answer_file.exists() else ""

    # chat.py result format: the record, prompt_id as a list, and one dialogue.
    result = dict(record)
    result["prompt_id"] = [record["prompt_id"]]
    result["dialogues"] = [{
        "prompt_id": record["prompt_id"],
        "question": record["prompt"],
        "model_response": answer,
    }]
    result["parity"] = {
        "task_id": tid,
        "agent": "claude-code",
        "model": args.model,
        "status": status,
        "elapsed_sec": round(time.time() - start, 1),
    }
    (out / "responses" / f"{tid}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, help="Claude model, passed as ANTHROPIC_MODEL")
    p.add_argument("--out", required=True, type=Path, help="Run directory")
    p.add_argument("--tiers", default="t1,t2,t3", help="Comma-separated: t1,t2,t3")
    p.add_argument("--task-ids", type=Path, help="File with one task id per line (subset)")
    p.add_argument("--limit", type=int, default=0, help="First N selected tasks; 0 = all")
    p.add_argument("--workers", type=int, default=4, help="Tasks run in parallel")
    p.add_argument("--image", default="finsearchcomp-parity-claude", help="Agent image")
    p.add_argument("--timeout", type=float, default=3600.0, help="Agent timeout per task (s)")
    p.add_argument("--pass-env", default=DEFAULT_PASS_ENV,
                   help="Comma-separated env vars passed to the agent container")
    p.add_argument("--overwrite", action="store_true", help="Rerun tasks that already have a response")
    args = p.parse_args()

    tiers = {t.strip().lower() for t in args.tiers.split(",") if t.strip()}
    tasks = load_tasks(tiers)
    if args.task_ids:
        wanted = {line.strip() for line in args.task_ids.read_text().splitlines() if line.strip()}
        tasks = [r for r in tasks if task_id(r) in wanted]
    if args.limit:
        tasks = tasks[: args.limit]

    out = args.out.resolve()
    (out / "responses").mkdir(parents=True, exist_ok=True)
    done = {f.stem for f in (out / "responses").glob("*.json")}
    todo = [r for r in tasks if args.overwrite or task_id(r) not in done]
    pass_env = [n.strip() for n in args.pass_env.split(",") if n.strip()]
    print(f"{len(tasks)} tasks selected, {len(todo)} to run, {args.workers} workers", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, r, args, out, pass_env): r for r in todo}
        for n, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            info = res["parity"]
            print(f"[{n}/{len(todo)}] {info['task_id']}: {info['status']} {info['elapsed_sec']}s", flush=True)

    # Combined chat.json for eval.py, in task-selection order.
    ids = [task_id(r) for r in tasks]
    combined = [
        json.loads((out / "responses" / f"{tid}.json").read_text(encoding="utf-8"))
        for tid in ids if (out / "responses" / f"{tid}.json").exists()
    ]
    (out / "chat.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(combined)} responses to {out / 'chat.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
