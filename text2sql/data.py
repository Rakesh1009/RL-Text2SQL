import os
import random
import re
from datasets import load_dataset

DB_ROOT = "/content/drive/MyDrive/RL_Text2SQL_storage/spider_db/database"


def parse_schema(db_id):
    schema_path = os.path.join(DB_ROOT, db_id, "schema.sql")

    if not os.path.exists(schema_path):
        return "Schema not found."

    tables = {}
    foreign_keys = []

    with open(schema_path, "r") as f:
        content = f.read()

    # Extract CREATE TABLE blocks
    create_blocks = re.findall(
        r"create table\s+(\w+)\s*\((.*?)\);",
        content,
        flags=re.IGNORECASE | re.DOTALL
    )

    for table_name, body in create_blocks:
        columns = []
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


def format_prompt(example):
    question = example["question"]
    db_id = example["db_id"]

    schema_text = parse_schema(db_id)

    prompt = f"""You are a SQL generator.

Database Schema:
{schema_text}

Question:
{question}

Generate ONLY the SQL query.
Do not explain anything.
"""

    return prompt


def load_spider_splits(train_size=400, val_size=50, seed=42):
    dataset = load_dataset("spider", split="train")

    random.seed(seed)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]

    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)

    def build_data(split_dataset):
        data = []
        for ex in split_dataset:
            data.append({
                "prompt": format_prompt(ex),
                "gold_sql": ex["query"],
                "db_id": ex["db_id"]
            })
        return data

    return build_data(train_dataset), build_data(val_dataset)
