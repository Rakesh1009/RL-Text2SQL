import yaml
from text2sql.data import load_spider_split
from text2sql.model import load_model
from text2sql.trainer import train


def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":

    config = load_config()
    mode = config.get("mode", "debug")

    print(f"Running in mode: {mode}")

    if mode == "debug":
        subset_size = config["training"]["subset_size"]
        train_data = load_spider_split(split="train", limit=subset_size)

        print("\nSample Example:\n")
        print(train_data[0]["prompt"])
        print("Gold SQL:", train_data[0]["gold_sql"])
        print("DB ID:", train_data[0]["db_id"])
        print("\nDebug pipeline validated.")

    elif mode == "train":
        subset_size = config["training"]["subset_size"]

        print("Loading dataset...")
        train_data = load_spider_split(split="train", limit=subset_size)

        print("Loading model...")
        model, tokenizer = load_model()

        print("Starting training...")
        train(model, tokenizer, train_data, config)

    else:
        raise ValueError("Invalid mode in config.")
