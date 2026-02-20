import os
import random
import re
from datasets import load_dataset
from tqdm import tqdm
import subprocess
import shutil
import sqlite3


def ensure_spider_db(db_path):
    if os.path.exists(db_path):
        return

    print(f"Spider database not found at {db_path}. Downloading...")
    # 1. Install Git LFS
    subprocess.run(["git", "lfs", "install"], check=True)

    # 2. Clone to a temporary folder
    temp_dir = "./spider_temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    subprocess.run(["git", "clone", "https://huggingface.co/datasets/prem-research/spider", temp_dir], check=True)

    # 3. Move contents and clean up
    os.makedirs(db_path, exist_ok=True)
    # The database folder inside the repo needs to be copied verbatim
    repo_db_dir = os.path.join(temp_dir, "database")
    if os.path.exists(repo_db_dir):
        shutil.copytree(repo_db_dir, db_path, dirs_exist_ok=True)
    else:
        print("Warning: database directory not found in the cloned repo!")
        
    shutil.rmtree(temp_dir)
    print("Spider database downloaded successfully.")


def parse_schema(db_path, db_id):
    sqlite_path = os.path.join(db_path, db_id, f"{db_id}.sqlite")
    schema_path = os.path.join(db_path, db_id, "schema.sql")
    
    content = ""
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            content = f.read()
    elif os.path.exists(sqlite_path):
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        tables_sql = cursor.fetchall()
        for t in tables_sql:
            if t[0]:
                content += t[0] + ";\n"
        conn.close()
    else:
        print(f"Schema/SQLite not found for: {db_id}")
        return "Schema not found."

    tables = {}
    foreign_keys = []

    # Extract CREATE TABLE blocks
    create_blocks = re.findall(
        r"create table\s+`?\"?(\w+)`?\"?\s*\((.*?)\);",
        content,
        flags=re.IGNORECASE | re.DOTALL
    )

    for table_name, body in create_blocks:
        columns = []
        # split by commas but we would need to be careful about commas inside parentheses,
        # but the simple split by comma generally works for most spider sqls unless there are functions.
        lines = body.split(",")

        for line in lines:
            line = line.strip()

            # Skip constraint-only lines
            if line.lower().startswith("foreign key"):
                fk_match = re.search(
                    r'foreign key\s*\("?(.*?)"?\)\s*references\s*`?(\w+)`?\("?(.*?)"?\)',
                    line,
                    flags=re.IGNORECASE
                )
                if fk_match:
                    src_col, ref_table, ref_col = fk_match.groups()
                    foreign_keys.append(
                        f"{table_name}.{src_col} → {ref_table}.{ref_col}"
                    )
                continue

            if line.lower().startswith("primary key"):
                continue

            # Extract column name
            col_match = re.match(r'"?(\w+)"?\s+', line)
            if col_match:
                col_name = col_match.group(1)

                if "primary key" in line.lower():
                    columns.append(f"{col_name} (primary key)")
                else:
                    columns.append(col_name)

        tables[table_name] = columns

    # Format clean schema text
    schema_text = ""

    for table, cols in tables.items():
        schema_text += f"Table {table}:\n"
        for col in cols:
            schema_text += f"  - {col}\n"
        schema_text += "\n"

    if foreign_keys:
        schema_text += "Foreign Keys:\n"
        for fk in foreign_keys:
            schema_text += f"  - {fk}\n"

    return schema_text


def format_prompt(db_path, example):
    question = example["question"]
    db_id = example["db_id"]

    schema_text = parse_schema(db_path, db_id)

    prompt = f"""You are a SQL generator.

Database Schema:
{schema_text}

Question:
{question}

Generate ONLY the SQL query.
Do not explain anything.
"""

    return prompt


def load_spider_splits(db_path, train_size=400, val_size=50, test_size=50, seed=42):
    ensure_spider_db(db_path)
    dataset = load_dataset("spider", split="train")

    random.seed(seed)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    
    test_start = train_size + val_size
    test_indices = indices[test_start:test_start + test_size]

    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    test_dataset = dataset.select(test_indices)

    def build_data(split_dataset, desc="Building data"):
        data = []
        for ex in tqdm(split_dataset, desc=desc):
            data.append({
                "prompt": format_prompt(db_path, ex),
                "gold_sql": ex["query"],
                "db_id": ex["db_id"]
            })
        return data

    print("Building train data...")
    t_data = build_data(train_dataset, desc="Train data")
    print("Building val data...")
    v_data = build_data(val_dataset, desc="Val data")
    print("Building test data...")
    test_data = build_data(test_dataset, desc="Test data")
    
    print("load_spider_splits finished!")
    return t_data, v_data, test_data
