import sqlite3
import time
import os

DB_ROOT = "/content/drive/MyDrive/RL_Text2SQL_storage/spider/database"


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


def execution_match(pred_sql, gold_sql, db_id):
    pred = execute_sql(pred_sql, db_id)
    gold = execute_sql(gold_sql, db_id)

    if not pred["success"] or not gold["success"]:
        return False, pred, gold

    return pred["result"] == gold["result"], pred, gold
