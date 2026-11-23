# Parity harness for the Harbor adapter

Upstream answers each question with one model call plus a search tool provided by an
internal gateway, which cannot be run outside it. For parity with the
[Harbor adapter](https://github.com/harbor-framework/adapters/tree/main/src/finsearchcomp),
both sides run the same CLI agent (Claude Code, same version) on the same instruction
and are graded by the same judge. Only the harness differs.

| File | Role |
| --- | --- |
| `Dockerfile` | Agent image: `python:3.12-slim` + Claude Code (`--build-arg CLAUDE_CODE_VERSION=...`) |
| `instruction.md` | The Harbor task instruction: upstream prompt plus where to write the answer |
| `run_agent.py` | Runs the agent per question in Docker, writes `chat.json` in `chat.py`'s format |
| `score.py` | Summarises `eval.py` output with the Harbor verifier's rules |

Grading uses the unchanged `finsearchcomp/eval/eval.py`. A judge name with a `/`
(e.g. `openai/gpt-5.4`) goes through LiteLLM (`models/litellm_api.py`).

## Run

```bash
pip install -r finsearchcomp/requirements.txt litellm
docker build -t finsearchcomp-parity-claude parity

export ANTHROPIC_API_KEY=...   # agent; add other variables with --pass-env
export OPENAI_API_KEY=...      # judge

python parity/run_agent.py --model claude-haiku-5-5 --tiers t2,t3 --out parity/runs/run1 --workers 6
PYTHONPATH=. python finsearchcomp/eval/eval.py --model_name openai/gpt-5.4 --workers 6 \
  --input parity/runs/run1/chat.json --output parity/runs/run1/eval.json
python parity/score.py parity/runs/run1/eval.json
```

`run_agent.py` skips tasks that already have a response, so an interrupted run can be
resumed with the same command. Use `--tiers`, `--task-ids` or `--limit` for a subset.

Run T1 on the same UTC day as the Harbor side: `eval.py` refreshes its market
snapshots when they are not from today.
