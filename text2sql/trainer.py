import torch
import torch.nn.functional as F
from text2sql.reward import compute_reward
from tqdm import tqdm
import random
import numpy as np
from text2sql.policy import reinforce_step, grpo_step
from text2sql.executor import execution_report


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate(model, tokenizer, dataset, max_new_tokens):

    model.eval()
    device = next(model.parameters()).device

    correct = 0

    with torch.no_grad():
        for example in dataset:

            prompt = example["prompt"]
            gold_sql = example["gold_sql"]
            db_id = example["db_id"]

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

            gen_tokens = generated[:, input_len:]
            pred_text = tokenizer.decode(
                gen_tokens[0],
                skip_special_tokens=True
            ).strip()

            report = execution_report(pred_text, gold_sql, db_id)

            if report["correct_result"]:
                correct += 1

    return correct / len(dataset)


def train(model, tokenizer, train_data, val_data, config):

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"])
    )

    epochs = config["training"]["epochs"]
    max_new_tokens = config["training"]["max_new_tokens"]

    rl_cfg = config.get("rl", {})
    algorithm = rl_cfg.get("algorithm", "reinforce")
    group_size = rl_cfg.get("group_size", 2)
    temperature = rl_cfg.get("temperature", 1.0)
    top_p = rl_cfg.get("top_p", 0.9)

    for epoch in range(epochs):

        model.train()
        total_reward = 0
        total_loss = 0

        print(f"\n===== Epoch {epoch} =====")

        for step, example in enumerate(tqdm(train_data)):

            prompt = example["prompt"]
            gold_sql = example["gold_sql"]
            db_id = example["db_id"]

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            if algorithm == "reinforce":
                loss, reward = reinforce_step(
                    model,
                    tokenizer,
                    inputs,
                    gold_sql,
                    db_id,
                    max_new_tokens,
                    temperature,
                    top_p,
                    debug=(step < 2)
                )

            elif algorithm == "grpo":
                loss, reward = grpo_step(
                    model,
                    tokenizer,
                    inputs,
                    gold_sql,
                    db_id,
                    max_new_tokens,
                    group_size,
                    temperature,
                    top_p,
                    debug=(step < 2)
                )

            else:
                raise ValueError("Unknown RL algorithm")

            total_reward += reward
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy = total_reward / len(train_data)
        avg_loss = total_loss / len(train_data)

        print(f"\nTrain Avg Reward: {train_accuracy}")
        print(f"Average Loss: {avg_loss}")

        val_accuracy = evaluate(
            model,
            tokenizer,
            val_data,
            max_new_tokens
        )

        print(f"Validation Avg Reward: {val_accuracy}")

