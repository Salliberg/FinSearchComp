import argparse
import datetime as dt
import json
import logging
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict

import akshare as ak

# Add project root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config.config_wrapper import initialize_config
from models.openai_api import load_model
from logger import get_logger
from finsearchcomp.data import fetch_single, get_snapshot_directory

# Setup logging
logger = get_logger(__name__)

# Default error score
DEFAULT_ERROR_SCORE = -100000
_SNAPSHOT_FRESHNESS_CHECKED = False


def _latest_snapshot_date(snapshot_dir: str) -> dt.date | None:
    """Return the most recent snapshot date parsed from filenames in the directory."""
    pattern = re.compile(r"_(\d{8})T\d+Z\.(csv|json)$", re.IGNORECASE)
    latest: dt.date | None = None

    try:
        entries = [
            entry for entry in os.listdir(snapshot_dir)
            if os.path.isfile(os.path.join(snapshot_dir, entry))
        ]
    except FileNotFoundError:
        logger.info("Snapshot directory %s does not exist yet.", snapshot_dir)
        return None

    for entry in entries:
        match = pattern.search(entry)
        if not match:
            continue
        candidate = dt.datetime.strptime(match.group(1), "%Y%m%d").date()
        if not latest or candidate > latest:
            latest = candidate
    return latest


def _refresh_snapshots() -> bool:
    """Invoke the snapshot fetch script to refresh on-disk market data."""
    script_path = os.path.join(root_dir, "scripts", "fetch_market_data.py")
    if not os.path.exists(script_path):
        logger.error("Snapshot refresh script not found at %s", script_path)
        return False

    logger.info("Refreshing market snapshots via %s", script_path)
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Snapshot refresh failed with exit code %s", exc.returncode)
        return False
    except OSError as exc:
        logger.error("Failed to execute snapshot refresh script: %s", exc)
        return False

    return True


def ensure_snapshots_current() -> None:
    """Ensure on-disk market snapshots are from the current calendar day."""
    global _SNAPSHOT_FRESHNESS_CHECKED
    if _SNAPSHOT_FRESHNESS_CHECKED:
        return

    snapshot_dir = get_snapshot_directory()
    today = dt.datetime.now().date()
    latest_date = _latest_snapshot_date(snapshot_dir)

    if latest_date != today:
        logger.info(
            "Cached snapshot date %s does not match today's date %s; refreshing.",
            latest_date,
            today,
        )
        if _refresh_snapshots():
            latest_date = _latest_snapshot_date(snapshot_dir)
        else:
            logger.warning("Snapshot refresh attempt failed; proceeding with existing data.")

    if latest_date != today:
        logger.warning(
            "Latest snapshot date %s still differs from today %s after refresh.",
            latest_date,
            today,
        )
    else:
        logger.info("Snapshot data confirmed for %s.", today)

    _SNAPSHOT_FRESHNESS_CHECKED = True


SCORE_KEYS = ("answer_score", "score")  # T2/T3 judges emit answer_score, T1 judges emit score


def _unwrap_score(value):
    """Accept a flat score (as the judge prompts specify) or the nested [[score]] form."""
    while isinstance(value, list) and value:
        value = value[0]
    return value


def parse_judge_output(judge_output: str) -> float:
    """Parse numerical score from judge's output.

    The judge prompts ask for {"answer_score": <0|1>} (T2/T3) or {"score": <0|1>} (T1),
    usually inside a plain ``` fence. Returns DEFAULT_ERROR_SCORE when no score is found,
    and None when a T1 judge reports {"score": "null"} (no real-time data available).
    """
    if not isinstance(judge_output, str):
        logger.warning(f"Input passed to parse_judge_output is not a string: {type(judge_output)}")
        return DEFAULT_ERROR_SCORE

    # Fenced blocks (with or without the `json` tag, closed by ``` or ``) first, then bare objects.
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*`{2,3}", judge_output, re.DOTALL)
    candidates += re.findall(r"\{[^{}]*\"(?:answer_score|score)\"[^{}]*\}", judge_output)

    # The final verdict comes last; earlier objects may be quoted examples.
    for raw in reversed(candidates):
        try:
            judge_json = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(judge_json, dict):
            continue
        for key in SCORE_KEYS:
            if key not in judge_json:
                continue
            value = _unwrap_score(judge_json[key])
            if isinstance(value, str) and value.strip().lower() == "null":
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                break

    logger.warning(f"Unable to find a score in judge output: {judge_output}")
    return DEFAULT_ERROR_SCORE


def load_judge(model_name):
    """A provider/model string (e.g. openai/gpt-5-mini) goes through LiteLLM."""
    if "/" in model_name:
        from models.litellm_api import load_model as load_litellm
        return load_litellm(model_name)
    return load_model(model_name)


def is_valid_score(score) -> bool:
    return score is not None and score != DEFAULT_ERROR_SCORE


def evaluate_response(
        judge_user_input,
        judge_model, 
        judge_system_prompt) -> Dict[str, Any]:
    """Evaluate model response"""

    # Call judge model
    history = [{"role": "system", "content": judge_system_prompt}]
    judge_response = judge_model([judge_user_input], [history])[0]
    
    # Parse score
    score = parse_judge_output(judge_response)
    
    # Return evaluation result
    return {
        "judge_response": judge_response,
        "score": score
    }


def get_judge_user_input(data, response):
    prompt = data['prompt']
    prompt_id = data['prompt_id'][0]
    prompt_template = data['judge_prompt_template']
    response_reference = data['response_reference']
    if 'T1' in prompt_id:
        akshare_ticker = data['akshare_ticker']
        tag = data['tags']
        method = data.get('method', None)
        ensure_snapshots_current()
        try:
            ground_truth = fetch_single(akshare_ticker, tag, method)
        except ValueError as exc:
            # e.g. an unsupported tag/method pair: no ground truth, so the judge
            # answers "null" and the item is left out of accuracy.
            logger.warning(f"No ground truth for {data['prompt_id']}: {exc}")
            ground_truth = None

        judge_user_input = prompt_template.format(
            prompt = prompt,
            response_reference = response_reference,
            ground_truth = ground_truth,
            response = response,
        )
    else:
        judge_user_input = prompt_template.format(
            prompt = prompt,
            response_reference = response_reference,
            response = response,
        )
    return judge_user_input


# Load input file and complete model evaluation
def process_file(data: Dict, model, output_file: str = None):
    """Process input file, evaluate model response"""
    prompt_id = data['prompt_id']
    
    # Initialize result
    results = dict([(k, v) for k,v in data.items()])
    results['evaluations'] = []
    results['accuracy'] = 0.0

    # Process each dialogue to be evaluated
    total_score = 0.0
    valid_evals = 0
    
    # Calculate the score of non-TS tasks
    non_ts_total_score = 0.0
    non_ts_valid_evals = 0
    
    logger.info(f"Data keys: {list(data.keys())}")
    logger.info(f"prompt_ids: {prompt_id}")
    logger.info(f"response_reference: {len(data.get('response_reference', []))}")
    logger.info(f"dialogues: {len(data.get('dialogues', []))}")
    
    for dialogue in data.get("dialogues", []):
        prompt_id = dialogue.get("prompt_id", "")
        
        # If this prompt_id is in eval_id list, then evaluation is needed
        prompt = data["prompt"]
        model_response = dialogue["model_response"]
        judge_system_prompt = data["judge_system_prompt"]
        judge_user_input = get_judge_user_input(data, model_response)


        # If necessary information is missing, skip this evaluation
        if not model_response:
            logger.warning(f"Dialog {prompt_id} lacks necessary evaluation information, skipping")
            continue
        
        # Evaluate response
        evaluation = evaluate_response(judge_user_input, model, judge_system_prompt)
        
        # Record evaluation result
        eval_result = {
            "id": prompt_id,
            "question": prompt,
            "model_response": model_response,
            "judge_user_input": judge_user_input,
            "judge_response": evaluation["judge_response"],
            "score": evaluation["score"]
        }
        
        results["evaluations"].append(eval_result)
        score = evaluation["score"]
        if not is_valid_score(score):
            # Unparseable judge output, or T1 "null" (no real-time data): exclude from accuracy.
            logger.warning(f"Evaluation {prompt_id} has no valid score ({score}); excluded")
            continue

        total_score += score
        valid_evals += 1
        if 'T1' not in str(prompt_id):
            non_ts_total_score += score
            non_ts_valid_evals += 1

        logger.info(f"Evaluation completed {prompt_id}: score {score}")
    
    # Calculate total accuracy
    results["valid"] = valid_evals > 0
    if valid_evals > 0:
        # Accuracy over all evaluations of this item (T1 included).
        results["accuracy"] = total_score / valid_evals
        # Accuracy over non-time-sensitive evaluations only (kept for reference).
        if non_ts_valid_evals > 0:
            results["non_ts_accuracy"] = non_ts_total_score / non_ts_valid_evals
    

    # Save result
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation results saved to {output_file}")
        logger.info(f"Total accuracy: {results['accuracy']:.4f}")
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evaluate model response')
    parser.add_argument('--model_name', type=str, default='deepseek-chat', help='Model name')
    parser.add_argument('--api_key', type=str, help='API key (optional, will override config file)')
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--output', type=str, default='result/eval.json', help='Output file path')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--workers', type=int, default=1, help='Items judged in parallel')
    args = parser.parse_args()
    
    # Build full path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    config_path = os.path.join(base_dir, args.config)
    
    # Initialize config
    try:
        initialize_config(config_path)
        logger.info(f"Config loaded from {config_path}")
    except Exception as e:
        logger.error(f"Error initializing config: {e}")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output file: {output_path}")
    
    # Load model
    model = load_judge(args.model_name)

    # Read entire JSON file instead of reading line by line
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        # Check if data is a list
        if isinstance(data, list):
            logger.info(f"Processing a list of {len(data)} dialogue objects")
            all_results = []
            total_accuracy_sum = 0.0
            valid_results = 0
            
            # Refresh T1 snapshots once, before any parallel work.
            if any('T1' in str(item.get('prompt_id')) for item in data):
                ensure_snapshots_current()

            if args.workers > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    processed = list(pool.map(lambda item: process_file(item, model, None), data))
            else:
                processed = (process_file(item, model, None) for item in data)

            for i, result in enumerate(processed):
                all_results.append(result)
                
                # Calculate overall accuracy over every item with a valid score (0 included)
                if result.get("valid"):
                    total_accuracy_sum += result["accuracy"]
                    valid_results += 1
                    
                logger.info(f"Processed item {i+1}/{len(data)}, accuracy: {result.get('accuracy', 0):.4f}")
        # Calculate overall accuracy across all items
            overall_accuracy = total_accuracy_sum / valid_results if valid_results > 0 else 0.0
            
            # Save all results to the output file
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as out_f:
                    json.dump(all_results, out_f, ensure_ascii=False, indent=2)
                    
                logger.info(f"All results saved to {output_path}")
                logger.info(f"Overall accuracy across all items: {overall_accuracy:.4f}")
        else:
            # Process single JSON object
            result = process_file(data, model, output_path)
            logger.info(f"Evaluation completed, total accuracy: {result['accuracy']:.4f}")


if __name__ == "__main__":
    main()
