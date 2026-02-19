from datasets import load_dataset


def format_prompt(example):
    question = example["question"]
    db_id = example["db_id"]

    prompt = f"""You are an expert SQL generator.

Database ID: {db_id}

Question:
{question}

Generate the correct SQL query.
"""
    return prompt


def load_spider_split(split="train", limit=None):
    dataset = load_dataset("spider", split=split)

    if limit is not None:
        dataset = dataset.select(range(limit))

    data = []
    for ex in dataset:
        data.append({
            "prompt": format_prompt(ex),
            "gold_sql": ex["query"],
            "db_id": ex["db_id"]
        })

    return data
