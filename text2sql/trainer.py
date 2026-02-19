import torch
import torch.nn.functional as F
from text2sql.reward import compute_reward
from tqdm import tqdm


def train(model, tokenizer, dataset, config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"]
    )

    epochs = config["training"]["epochs"]
    max_new_tokens = config["training"]["max_new_tokens"]

    for epoch in range(epochs):

        total_reward = 0

        for example in tqdm(dataset):

            prompt = example["prompt"]
            gold_sql = example["gold_sql"]
            db_id = example["db_id"]

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            # Generate with sampling
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

            # Compute logprob
            outputs = model(**generated, labels=generated)
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_reward = total_reward / len(dataset)
        print(f"Epoch {epoch} | Avg Reward: {avg_reward}")
