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


def compute_reward(pred_sql, gold_sql, db_id, config=None, return_details=False):

    # -----------------------
    # 1. Format Check
    # -----------------------
    if not check_format(pred_sql):
        if return_details:
             return -1.0, {"pred_executable": False, "gold_executable": False, "pred_result": [], "gold_result": [], "error": "Format Check Failed"}
        return -1.0

    report = execution_report(pred_sql, gold_sql, db_id)

    total_reward = 0.0

    # -----------------------
    # 2. Execution Reward
    # -----------------------
    if not report["pred_executable"]:
        if return_details:
            return -0.5, report
        return -0.5

    total_reward += 1.0  # execution success

    # -----------------------
    # 3. Column + Row Scoring
    # -----------------------
    pred_rows = report.get("pred_result", []) or []
    gold_rows = report.get("gold_result", []) or []

    # If gold result empty → avoid division by zero
    if not gold_rows:
        if return_details:
            return float(total_reward), report
        return float(total_reward)

    # ---- Column score ----
    if len(pred_rows) > 0:
        pred_col_count = len(pred_rows[0])
        gold_col_count = len(gold_rows[0])
        col_score = min(pred_col_count, gold_col_count) / max(pred_col_count, gold_col_count)
    else:
        col_score = 0.0

    total_reward += 2.0 * col_score

    # ---- Row overlap score (unordered) ----
    pred_set = set(tuple(r) for r in pred_rows)
    gold_set = set(tuple(r) for r in gold_rows)

    if len(gold_set) > 0:
        row_overlap = len(pred_set & gold_set) / len(gold_set)
    else:
        row_overlap = 0.0

    total_reward += 2.0 * row_overlap

    # -----------------------
    # 4. Exact unordered match bonus
    # -----------------------
    if pred_set == gold_set and len(pred_set) > 0:
        total_reward += 0.5

        # -----------------------
        # 5. Order bonus
        # -----------------------
        if pred_rows == gold_rows:
            total_reward += 0.5

    if return_details:
        return float(total_reward), report
    return float(total_reward)
