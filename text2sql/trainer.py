import torch
import torch.nn.functional as F
from text2sql.reward import compute_reward
from tqdm import tqdm
import random
import numpy as np
import os
import json
import matplotlib.pyplot as plt
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


def train(model, ref_model, tokenizer, train_data, val_data, config):

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if ref_model is not None:
        ref_model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"])
    )

    training_cfg = config.get("training", {})
    epochs = training_cfg.get("epochs", 3)
    max_new_tokens = training_cfg.get("max_new_tokens", 128)
    resume_path = training_cfg.get("resume", False)

    rl_cfg = config.get("rl", {})
    algorithm = rl_cfg.get("algorithm", "reinforce")
    kl_beta = rl_cfg.get("kl_beta", 0.05)
    group_size = rl_cfg.get("group_size", 2)
    temperature = rl_cfg.get("temperature", 1.0)
    top_p = rl_cfg.get("top_p", 0.9)

    base_output_dir = config.get("paths", {}).get("checkpoints", "./checkpoints")
    os.makedirs(base_output_dir, exist_ok=True)

    # Calculate or extract Run Name
    if isinstance(resume_path, str) and os.path.exists(resume_path):
        run_name = os.path.basename(os.path.dirname(os.path.normpath(resume_path)))
    else:
        existing_runs = [d for d in os.listdir(base_output_dir) if d.startswith("run_")]
        run_ids = [int(d.split("_")[1]) for d in existing_runs if d.split("_")[1].isdigit()]
        next_id = max(run_ids) + 1 if run_ids else 1
        run_name = f"run_{next_id}"

    output_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Logging to Checkpoint Directory: {output_dir} ---")

    history = {
        "train_loss": [],
        "train_reward": [],
        "val_reward": []
    }
    
    start_epoch = 0
    best_val_reward = float("-inf")

    if isinstance(resume_path, str) and os.path.exists(resume_path):
        opt_path = os.path.join(resume_path, "optimizer.pt")
        hist_path = os.path.join(resume_path, "history.json")
        
        if os.path.exists(opt_path):
            print(f"Loading optimizer state from {opt_path}")
            opt_state = torch.load(opt_path, map_location=device)
            optimizer.load_state_dict(opt_state)
            
        if os.path.exists(hist_path):
            with open(hist_path, "r") as f:
                history = json.load(f)
            start_epoch = len(history.get("train_loss", []))
            best_val_reward = max(history.get("val_reward", [float("-inf")]))
            print(f"Resuming from epoch {start_epoch}, previous best val: {best_val_reward:.4f}")

    for epoch in range(start_epoch, epochs):

        model.train()
        total_reward = 0
        total_loss = 0

        print(f"\n===== Epoch {epoch} =====")

        pbar = tqdm(train_data, desc=f"Epoch {epoch}")
        for step, example in enumerate(pbar):

            prompt = example["prompt"]
            gold_sql = example["gold_sql"]
            db_id = example["db_id"]

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            if algorithm == "reinforce":
                loss, reward = reinforce_step(
                    model,
                    ref_model,
                    tokenizer,
                    inputs,
                    gold_sql,
                    db_id,
                    max_new_tokens,
                    temperature,
                    top_p,
                    kl_beta,
                    debug=(step < 2)
                )

            elif algorithm == "grpo":
                loss, reward = grpo_step(
                    model,
                    ref_model,
                    tokenizer,
                    inputs,
                    gold_sql,
                    db_id,
                    max_new_tokens,
                    group_size,
                    temperature,
                    top_p,
                    kl_beta,
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

        history["train_loss"].append(avg_loss)
        history["train_reward"].append(train_accuracy)
        history["val_reward"].append(val_accuracy)

        # Save last checkpoint
        last_dir = os.path.join(output_dir, "last")
        model.save_pretrained(last_dir)
        tokenizer.save_pretrained(last_dir)
        torch.save(optimizer.state_dict(), os.path.join(last_dir, "optimizer.pt"))
        
        with open(os.path.join(last_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
            
        print(f"Saved 'last' checkpoint to: {last_dir}")
        
        # Save best checkpoint
        if val_accuracy > best_val_reward:
            best_val_reward = val_accuracy
            best_dir = os.path.join(output_dir, "best")
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            torch.save(optimizer.state_dict(), os.path.join(best_dir, "optimizer.pt"))
            
            with open(os.path.join(best_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)
                
            print(f"*** New best validation reward! Saved 'best' checkpoint to: {best_dir} ***")

        # Plot metrics
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        actual_epochs = len(history["train_loss"])
        plt.plot(range(actual_epochs), history["train_loss"], label="Train Loss", marker="o")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # Reward plot
        plt.subplot(1, 2, 2)
        plt.plot(range(actual_epochs), history["train_reward"], label="Train Reward", marker="o", color="blue")
        plt.plot(range(actual_epochs), history["val_reward"], label="Val Reward", marker="o", color="orange")
        plt.title("Avg Reward")
        plt.xlabel("Epoch")
        plt.ylabel("Reward Score")
        plt.legend()
        
        plot_path = os.path.join(output_dir, "training_metrics.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"\nTraining metrics graph saved to: {plot_path}")

