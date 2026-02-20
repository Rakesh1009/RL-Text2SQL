import torch
import torch.nn.functional as F
from text2sql.reward import compute_reward

running_baseline = 0.0
baseline_momentum = 0.9


# =========================================================
# REINFORCE
# =========================================================
def reinforce_step(
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
    debug=False
):
    global running_baseline

    device = next(model.parameters()).device
    input_len = inputs["input_ids"].shape[1]

    # -----------------------------
    # 1️⃣ Sample from policy
    # -----------------------------
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

    reward = float(compute_reward(pred_text, gold_sql, db_id))

    if debug:
        prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        print("\n" + "=" * 50)
        print("--- REINFORCE Debug ---")
        print("\n[PROMPT]")
        print(prompt_text.strip())
        print("\n[GOLD SQL]")
        print(gold_sql)
        print("\n[GENERATED SQL]")
        print(pred_text)
        print(f"\n[Reward]: {reward}")
        print("=" * 50 + "\n")

    # -----------------------------
    # 2️⃣ Computing Policy log-probs
    # -----------------------------
    outputs = model(input_ids=generated)
    logits = outputs.logits

    # Slice strictly to generated tokens BEFORE softmax to save giant VRAM graphs
    gen_logits = logits[:, input_len - 1 : -1, :]
    gen_labels = generated[:, input_len:]

    log_probs = F.log_softmax(gen_logits, dim=-1)
    gen_log_probs = log_probs.gather(
        2,
        gen_labels.unsqueeze(-1)
    ).squeeze(-1)

    sequence_logprob = gen_log_probs.sum()
    # Normalize by generation length
    sequence_logprob = sequence_logprob / gen_tokens.shape[1]

    # -----------------------------
    # 2.5️⃣ Reference Model KL
    # -----------------------------
    if ref_model is not None:
        with torch.no_grad():
            ref_outputs = ref_model(input_ids=generated)
            ref_logits = ref_outputs.logits

        ref_gen_logits = ref_logits[:, input_len - 1 : -1, :]
        ref_log_probs = F.log_softmax(ref_gen_logits, dim=-1)
        ref_gen_log_probs = ref_log_probs.gather(
            2,
            gen_labels.unsqueeze(-1)
        ).squeeze(-1)

        kl_per_token = gen_log_probs - ref_gen_log_probs
        kl = kl_per_token.sum() / gen_tokens.shape[1]

        del ref_outputs
        del ref_logits
        del ref_gen_logits
    else:
        kl = torch.tensor(0.0, device=device)

    del outputs
    del logits
    del gen_logits
    del log_probs

    # -----------------------------
    # 3️⃣ Entropy bonus
    # -----------------------------
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(-1)
    gen_entropy = entropy[:, input_len:].mean()

    # -----------------------------
    # 4️⃣ Baseline & Advantage
    # -----------------------------
    running_baseline = (
        baseline_momentum * running_baseline
        + (1 - baseline_momentum) * reward
    )

    advantage = reward - running_baseline
    advantage = torch.tensor(advantage, device=device).clamp(-10.0, 10.0)

    # -----------------------------
    # 5️⃣ Final Loss
    # -----------------------------
    entropy_bonus = 0.001 * gen_entropy
    loss = -advantage * sequence_logprob - entropy_bonus
    final_loss = loss + kl_beta * kl

    if debug:
        print(f"[REINFORCE Mean KL]: {kl.item():.6f}")

    return final_loss, reward


# =========================================================
# GRPO
# =========================================================
def grpo_step(
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
    debug=False
):
    device = next(model.parameters()).device
    input_len = inputs["input_ids"].shape[1]

    rewards = []
    logprobs = []
    kls = []
    texts = []

    for _ in range(group_size):

        # -----------------------------
        # 1️⃣ Sample
        # -----------------------------
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

        reward_val, details = compute_reward(pred_text, gold_sql, db_id, return_details=True)
        reward = float(reward_val)

        # -----------------------------
        # 2️⃣ Log-probs
        # -----------------------------
        outputs = model(input_ids=generated)
        logits = outputs.logits

        # Slice strictly to generated tokens BEFORE softmax to save massive VRAM overhead
        gen_logits = logits[:, input_len - 1 : -1, :]
        gen_labels = generated[:, input_len:]

        log_probs = F.log_softmax(gen_logits, dim=-1)
        gen_log_probs = log_probs.gather(
            2,
            gen_labels.unsqueeze(-1)
        ).squeeze(-1)

        sequence_logprob = gen_log_probs.sum()

        # Normalize by generation length
        sequence_logprob = sequence_logprob / gen_tokens.shape[1]

        # -----------------------------
        # 2.5️⃣ Reference Model KL
        # -----------------------------
        if ref_model is not None:
            with torch.no_grad():
                ref_outputs = ref_model(input_ids=generated)
                ref_logits = ref_outputs.logits

            ref_gen_logits = ref_logits[:, input_len - 1 : -1, :]
            ref_log_probs = F.log_softmax(ref_gen_logits, dim=-1)
            ref_gen_log_probs = ref_log_probs.gather(
                2,
                gen_labels.unsqueeze(-1)
            ).squeeze(-1)

            kl_per_token = gen_log_probs - ref_gen_log_probs
            kl = kl_per_token.sum() / gen_tokens.shape[1]
            del ref_outputs
            del ref_logits
            del ref_gen_logits
        else:
            kl = torch.tensor(0.0, device=device)

        rewards.append(reward)
        logprobs.append(sequence_logprob)
        kls.append(kl)
        texts.append(pred_text)
        
        # Keep details for debugging
        if debug:
            if "details_list" not in locals():
                details_list = []
            details_list.append(details)

        del outputs
        del logits
        del gen_logits
        del log_probs
        torch.cuda.empty_cache()

    # -----------------------------
    # 3️⃣ Advantage Normalization
    # -----------------------------
    rewards_tensor = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=device
    )

    advantages = rewards_tensor - rewards_tensor.mean()
    advantages = advantages / (rewards_tensor.std(unbiased=False) + 1e-8)
    advantages = advantages.clamp(-10.0, 10.0)

    # -----------------------------
    # 4️⃣ Policy Loss
    # -----------------------------
    final_loss = 0
    for adv, logprob, kl in zip(advantages, logprobs, kls):
        sample_loss = -adv * logprob + kl_beta * kl
        final_loss += sample_loss

    final_loss = final_loss / group_size
    avg_reward = rewards_tensor.mean().item()
    
    avg_kl = torch.stack(kls).mean().item()

    # -----------------------------
    # Debug Print
    # -----------------------------
    if debug:
        prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        print("\n" + "=" * 50)
        print("--- GRPO Debug ---")
        print("\n[PROMPT]")
        print(prompt_text.strip())
        print("\n[GOLD SQL]")
        print(gold_sql)
        print(f"\n[GRPO Mean KL]: {avg_kl:.6f}")
        for i in range(group_size):
            print(f"\n--- Sample {i} | Reward: {rewards[i]:.4f} | KL: {kls[i].item():.4f} ---")
            print("[GENERATED SQL]")
            print(texts[i].strip())
            
            det = details_list[i]
            if not det.get("pred_executable"):
                print(f"[EXECUTION ERROR] {det.get('pred_error', det.get('error', 'Execution Failed'))}")
            else:
                print(f"[RESULTS]")
                print(f"  Rows Returned: {len(det.get('pred_result', []))}")
                print(f"  Top 3 Rows: {det.get('pred_result', [])[:3]}")
                
            if i == 0:
                print("\n[GOLD RESULTS]")
                print(f"  Rows Returned: {len(det.get('gold_result', []))}")
                print(f"  Top 3 Rows: {det.get('gold_result', [])[:3]}")
            
        print("=" * 50 + "\n")

    return final_loss, avg_reward