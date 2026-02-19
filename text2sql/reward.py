from text2sql.executor import execution_match


def compute_reward(pred_sql, gold_sql, db_id):
    match, _, _ = execution_match(pred_sql, gold_sql, db_id)

    if match:
        return 1.0

    return -1.0
