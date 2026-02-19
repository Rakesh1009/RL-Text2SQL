import re
from text2sql.executor import execution_report


def check_format(pred_sql):
    """
    Basic format check:
    - Must start with SELECT
    - Must not contain obvious chatty phrases
    """

    if not pred_sql:
        return False

    sql = pred_sql.strip().lower()

    if not sql.startswith("select"):
        return False

    forbidden_phrases = [
        "sure",
        "here is",
        "explanation",
        "use the database",
        "please"
    ]

    for phrase in forbidden_phrases:
        if phrase in sql:
            return False

    return True


def compute_reward(pred_sql, gold_sql, db_id, config=None):
    """
    Layered reward:
    Format -> Execution -> Result -> Length
    """

    # Default weights
    weights = {
        "format": 1.0,
        "execution": 2.0,
        "result": 3.0,
        "length": 0.5
    }

    if config and "reward_weights" in config:
        weights.update(config["reward_weights"])

    total_reward = 0.0

    # ------------------------
    # 1. Format Reward
    # ------------------------
    format_ok = check_format(pred_sql)

    if format_ok:
        total_reward += weights["format"]
    else:
        total_reward -= weights["format"]
        return total_reward  # stop early if format bad

    # ------------------------
    # 2. Execution Reward
    # ------------------------
    report = execution_report(pred_sql, gold_sql, db_id)

    if report["pred_executable"]:
        total_reward += weights["execution"]
    else:
        total_reward -= weights["execution"]
        return total_reward  # stop if not executable

    # ------------------------
    # 3. Result Reward
    # ------------------------
    if report["correct_result"]:
        total_reward += weights["result"]
    else:
        total_reward -= weights["result"]

    # ------------------------
    # 4. Length Reward (optional bonus)
    # Only if result correct
    # ------------------------
    if report["correct_result"]:
        sql_length = len(pred_sql.split())
        if 5 <= sql_length <= 100:
            total_reward += weights["length"]

    return total_reward
