import yaml
from text2sql.data import load_spider_splits
from text2sql.model import load_model
from text2sql.trainer import train


def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":

    config = load_config()
    db_path = config.get("paths", {}).get("spider_db", "./data/spider_db")

    data_cfg = config.get("data", {})
    train_size = data_cfg.get("train_size", 20)
    val_size = data_cfg.get("val_size", 5)
    test_size = data_cfg.get("test_size", 5)

    print(f"Using Spider DB path: {db_path}")
    print("Loading dataset...")
    train_data, val_data, test_data = load_spider_splits(
        db_path=db_path,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=42
    )

    print("Loading model...")
    model, ref_model, tokenizer = load_model(config)

    print("Starting training...")
    train(model, ref_model, tokenizer, train_data, val_data, config)
