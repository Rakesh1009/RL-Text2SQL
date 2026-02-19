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


def load_spider_splits(train_size=400, val_size=50, seed=42):

    dataset = load_dataset("spider", split="train")
    dataset = dataset.shuffle(seed=seed)

    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))

    def convert(ds):
        data = []
        for ex in ds:
            data.append({
                "prompt": format_prompt(ex),
                "gold_sql": ex["query"],
                "db_id": ex["db_id"]
            })
        return data

    return convert(train_dataset), convert(val_dataset)
