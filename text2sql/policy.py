import torch
from text2sql.reward import compute_reward

running_baseline = 0.0
baseline_momentum = 0.9


def reinforce_step(model, tokenizer, inputs,
                   gold_sql, db_id,
                   max_new_tokens,
                   temperature,
                   top_p,
                   debug=False):

    device = next(model.parameters()).device
    input_len = inputs["input_ids"].shape[1]

    # Sample from policy
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id
        )

    gen_tokens = generated[:, input_len:]
    pred_text = tokenizer.decode(
        gen_tokens[0],
        skip_special_tokens=True
    ).strip()

    reward = compute_reward(pred_text, gold_sql, db_id)

    if debug:
        print("\n--- REINFORCE Debug ---")
        print("Predicted SQL:", pred_text)
        print("Reward:", reward)

    # Compute log-prob loss
    outputs = model(input_ids=generated)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = generated[:, 1:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    selected_log_probs = log_probs.gather(
        2,
        shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Only keep generated part
    gen_log_probs = selected_log_probs[:, input_len-1:]
    sequence_logprob = gen_log_probs.sum()

    global running_baseline

    running_baseline = (
        baseline_momentum * running_baseline
        + (1 - baseline_momentum) * reward
    )

    advantage = reward - running_baseline

    loss = -advantage * sequence_logprob

    return loss, reward


def grpo_step(model, tokenizer, inputs,
              gold_sql, db_id,
              max_new_tokens,
              group_size,
              temperature,
              top_p,
              debug=False):

    device = next(model.parameters()).device
    input_len = inputs["input_ids"].shape[1]

    rewards = []
    logprobs = []
    texts = []

    for _ in range(group_size):

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id
            )

        gen_tokens = generated[:, input_len:]
        pred_text = tokenizer.decode(
            gen_tokens[0],
            skip_special_tokens=True
        ).strip()

        reward = compute_reward(pred_text, gold_sql, db_id)

        # Forward pass for logprobs
        outputs = model(input_ids=generated)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = generated[:, 1:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        selected_log_probs = log_probs.gather(
            2,
            shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        gen_log_probs = selected_log_probs[:, input_len-1:]
        sequence_logprob = gen_log_probs.sum()

        rewards.append(reward)        # 🔥 YOU FORGOT THIS
        logprobs.append(sequence_logprob)
        texts.append(pred_text)

    rewards_tensor = torch.tensor(rewards, device=device)

    # Group baseline
    advantages = rewards_tensor - rewards_tensor.mean()

    final_loss = 0
    for adv, logprob in zip(advantages, logprobs):
        final_loss += -adv * logprob

    final_loss = final_loss / group_size
    avg_reward = rewards_tensor.mean().item()

    if debug:
        print("\n--- GRPO Debug ---")
        for i in range(group_size):
            print(f"Sample {i} | Reward={rewards[i]}")
            print(texts[i])
            print("----")

    return final_loss, avg_reward

