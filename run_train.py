import yaml
from text2sql.data import load_spider_splits
from text2sql.model import load_model
from text2sql.trainer import train


def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":

    config = load_config()

    print("Loading dataset...")
    train_data, val_data = load_spider_splits(
        train_size=20,
        val_size=5,
        seed=42
    )

    print("Loading model...")
    model, tokenizer = load_model()

    print("Starting training...")
    train(model, tokenizer, train_data, val_data, config)
