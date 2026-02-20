import sqlite3
import time
import os
import yaml

def get_db_root():
    try:
        with open("configs/default.yaml", "r") as f:
            config = yaml.safe_load(f)
            return config.get("paths", {}).get("spider_db", "./data/spider_db")
    except Exception:
        return os.getenv("SPIDER_DB_ROOT", "./data/spider_db")

DB_ROOT = get_db_root()


def execute_sql(sql, db_id):
    db_path = os.path.join(DB_ROOT, db_id, f"{db_id}.sqlite")

    if not os.path.exists(db_path):
        return {"success": False, "error": "DB not found", "result": None, "time": 0}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        start = time.time()
        cursor.execute(sql)
        result = cursor.fetchall()
        exec_time = time.time() - start

        conn.close()

        return {"success": True, "error": None, "result": result, "time": exec_time}

    except Exception as e:
        return {"success": False, "error": str(e), "result": None, "time": 0}


def execution_report(pred_sql, gold_sql, db_id):
    pred = execute_sql(pred_sql, db_id)
    gold = execute_sql(gold_sql, db_id)

    report = {
        "format_ok": True,      # handled separately in reward
        "pred_executable": pred["success"],
        "gold_executable": gold["success"],
        "correct_result": False,
        "pred_result": pred["result"],
        "gold_result": gold["result"],
        "pred_error": pred["error"],
        "execution_time": pred["time"]
    }

    if pred["success"] and gold["success"]:
        report["correct_result"] = pred["result"] == gold["result"]

    return report
