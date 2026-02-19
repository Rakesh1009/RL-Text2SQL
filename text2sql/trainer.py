import torch
import torch.nn.functional as F
from text2sql.reward import compute_reward
from tqdm import tqdm
import random
import numpy as np


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
                do_sample=False  # deterministic eval
            )

            gen_tokens = generated[:, input_len:]
            pred_text = tokenizer.decode(
                gen_tokens[0],
                skip_special_tokens=True
            ).strip()

            reward = compute_reward(pred_text, gold_sql, db_id)

            if reward == 1:
                correct += 1

    return correct / len(dataset)


def train(model, tokenizer, train_data, val_data, config):

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"])
    )

    epochs = config["training"]["epochs"]
    max_new_tokens = config["training"]["max_new_tokens"]

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
            input_len = inputs["input_ids"].shape[1]

            # Sampling generation
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9
                )

            gen_tokens = generated[:, input_len:]
            pred_text = tokenizer.decode(
                gen_tokens[0],
                skip_special_tokens=True
            ).strip()

            reward = compute_reward(pred_text, gold_sql, db_id)
            total_reward += reward

            # Debug print first 2 samples
            if step < 2:
                print("\n--- Debug Sample ---")
                print("Prompt:", prompt)
                print("Predicted SQL:", pred_text)
                print("Gold SQL:", gold_sql)
                print("Reward:", reward)

            # Compute logprob
            outputs = model(
                input_ids=generated,
                labels=generated
            )

            logits = outputs.logits

            shift_logits = logits[:, :-1, :]
            shift_labels = generated[:, 1:]

            log_probs = F.log_softmax(shift_logits, dim=-1)
            selected_log_probs = log_probs.gather(
                2,
                shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            gen_log_probs = selected_log_probs[:, input_len-1:]
            sequence_logprob = gen_log_probs.sum()

            loss = -reward * sequence_logprob
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy = total_reward / len(train_data)
        avg_loss = total_loss / len(train_data)

        print(f"\nTrain Execution Accuracy: {train_accuracy}")
        print(f"Average Loss: {avg_loss}")

        # Validation
        val_accuracy = evaluate(
            model,
            tokenizer,
            val_data,
            max_new_tokens
        )

        print(f"Validation Execution Accuracy: {val_accuracy}")
