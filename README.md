# RL-Text2SQL

This repository implements a Reinforcement Learning (RL) framework for fine-tuning Large Language Models (LLMs) on the Text-to-SQL task, primarily utilizing the robust Spider dataset. 

The project evaluates generated SQL queries directly against their target SQLite databases and updates the base model's logic through state-of-the-art RL paradigms like **GRPO** (Group Relative Policy Optimization) and **REINFORCE**.

---

## Process Flow

1. **Environment Initialization**: Automatically downloads and prepares the Spider dataset along with all accompanying `.sqlite` databases utilizing `git-lfs` to ensure a consistent data state.
2. **Model Bootstrapping**: Statically loads a pre-trained Causal Language Model (e.g., Qwen 2.5) entirely in 4-bit quantization. A lightweight Low-Rank Adaptation (LoRA) layer is injected into the attention modules to enable highly focused parameter updates.
3. **Trajectory Rollouts**: During each training step, the model acts as a policy, generating unconstrained SQL strings from natural language prompts.
4. **Environment Execution**: The generated SQL texts are stripped, formatted, and strictly run inside the corresponding SQLite environment database to obtain structured tables of rows and columns.
5. **Reward & Advantage Scoring**: The query's execution trace is heavily scrutinized using a highly granular reward function (see below), comparing both formatting and raw algebraic correctness against the "Gold" Target SQL response. 
6. **Policy Update**: The system evaluates the log-probabilities of the generated token trajectories and performs a backward step through an AdamW optimizer to natively modify the policy's weights, encouraging future generations to secure higher RL rewards.

---

## Optimizations

We have implemented aggressive software-layer constraints and hardware optimizations so the heavy RL loop can run smoothly natively or in local docker containers:

- **4-Bit NF4 Quantization**: Base models (and optional KL-Divergence reference models) are dynamically loaded via `BitsAndBytes` in strictly NF4 dual-quantized datatypes. This slashes the GPU VRAM overhead phenomenally.
- **LoRA-Driven Backpropagation**: Instead of fine-tuning the full weights of large models, gradients are passed exclusively through `Requires_Grad` LoRA adapter injections.
- **Aggressive Garbage Collection**: During roll-out logprob sequences, enormous computation graphs are immediately disposed of via sequential `del outputs` and synchronous `torch.cuda.empty_cache()` sweeps per iteration to strictly prevent OOM cascading on longer generation sequences.
- **Hermetic Docker Environment**: The training loop relies on `uv` to instantiate virtualized, portable dependency trees baked natively into a single Docker Compose execution graph, obliterating standard `pip` cross-contamination faults.

---

## Reward Structure

Text-to-SQL is fundamentally a functional problem, not just a semantic alignment objective. Thus, instead of blindly trusting automated cross-entropy matching, we employ an iterative, execution-verified scoring methodology:

1. **Format Check (-1.0)**: The generated SQL must explicitly contain syntax properties (starting with `SELECT` queries) and critically may not contain conversational padding (e.g., "Here is the explanation..."). Fails instantly without execution.
2. **Execution Verification (+1.0 / -0.5)**: The SQL string is executed locally. Valid SQLite syntax is rewarded (+1.0), whereas hallucinations or syntactically invalid table queries are heavily penalized (-0.5).
3. **Column Overlap (+2.0 Max)**: Measures the ratio of intersecting column constraints against the Gold reference standard.
4. **Row Overlap (+2.0 Max)**: Computes the exact distinct intersection bounds for identically returned table rows, mathematically checking mathematical and filtering (WHERE) accuracy.
5. **Set Equivalence Bonus (+0.5)**: Unlocks if the output set identically captures every single row required mathematically, regardless of `ORDER BY` formatting.
6. **Strict Output Ordering (+0.5)**: Granted if the list of rows identically shadows the exact returned list of the Gold query perfectly, validating absolute query sorting accuracy.

---

## Policy & Regularization

The framework abstracts its training loop so you can hot-swap policy architectures rapidly via `configs/default.yaml`. 

### GRPO (Group Relative Policy Optimization)
Instead of relying on a completely secondary "Critic / Value Model" taking up memory footprint, GRPO computes advantage scales dynamically. By rolling out `group_size` samples against the same prompt horizontally, evaluating their SQLite accuracy independently, and normalizing those resulting scores internally (`adv = reward - mean / std`), the system gracefully grounds its RL gradient scaling directly into its relative iterative outputs!

### Constrained Policy Optimization (KL Regularization)
To guarantee the LoRA model doesn't over-fit grammatically to the database structures or suffer RL mode collapse ("jailbreaking" random tokens into SQLite environments), the system heavily utilizes Kullback-Leibler (KL) divergence penalization.

If `use_kl: true` is configured:
- An immaculate, identically quantized frozen **Reference Model** is dynamically instantiated into VRAM.
- As the policy generates tokens, the identical sequences are evaluated asynchronously by the Reference model.
- An additive penalty is dynamically calculated mathematically per-token:
```python
sample_loss = -advantage * policy_logprob + kl_beta * kl
```
This forces the fine-tuned model to safely anchor itself directly into the foundational grammatical probability structures of its pre-training domain.

## Checkpointing & Metrics

The training script automatically manages its own internal state, seamlessly structuring outputs by numerical run directories (`./checkpoints/run_1/`, `./checkpoints/run_2/`).

To prevent massive storage bloat, checkpoints aggressively overwrite onto two static targets inside their run directory:
1. `last/`: Saves the raw optimizer state, `history.json`, and LoRA adapter weights unconditionally at the end of every single epoch.
2. `best/`: Only overwrites and saves if the validation reward achieves a brand-new high score.

**Dynamic Graphing**: A dual-axis `matplotlib` graph plotting `Training Loss` alongside both `Training Reward` and `Validation Reward` is automatically drawn and overwritten dynamically onto your disk as `training_metrics.png` the very moment an epoch concludes.
**Resuming**: Pointing `resume: "./checkpoints/run_1/last"` inside `configs/default.yaml` will seamlessly load the previous `history.json`, intercept the `optimizer.pt`, mount the LoRA checkpoints, and autonomously pick up generating its metrics graphs exactly where it left off!

---

## Example Run Trace

Below is an example snippet of the GRPO output printing to the terminal during Docker Compose execution, showcasing the model generating, evaluating, and penalizing/rewarding dynamically based on its SQL performance:

```text
rl-text2sql-train  | ==================================================
rl-text2sql-train  | --- GRPO Debug ---
rl-text2sql-train  | 
rl-text2sql-train  | [PROMPT]
rl-text2sql-train  | You are a SQL generator.
rl-text2sql-train  | 
rl-text2sql-train  | Database Schema:
rl-text2sql-train  | Table county:
rl-text2sql-train  |   - County_Id
rl-text2sql-train  |   - County_name
rl-text2sql-train  |   - Population
rl-text2sql-train  |   - Zip_code
rl-text2sql-train  | 
rl-text2sql-train  | Table party:
rl-text2sql-train  |   - Party_ID
rl-text2sql-train  |   - Year
rl-text2sql-train  |   - Party
rl-text2sql-train  |   - Governor
rl-text2sql-train  |   - Lieutenant_Governor
rl-text2sql-train  |   - Comptroller
rl-text2sql-train  |   - Attorney_General
rl-text2sql-train  |   - US_Senate
rl-text2sql-train  | 
rl-text2sql-train  | Table election:
rl-text2sql-train  |   - Election_ID
rl-text2sql-train  |   - Counties_Represented
rl-text2sql-train  |   - District
rl-text2sql-train  |   - Delegate
rl-text2sql-train  |   - Party
rl-text2sql-train  |   - First_Elected
rl-text2sql-train  |   - Committee
rl-text2sql-train  | 
rl-text2sql-train  | Foreign Keys:
rl-text2sql-train  |   - election.`Party` → party.`Party_ID`
rl-text2sql-train  |   - election.`District` → county.`County_Id`
rl-text2sql-train  | 
rl-text2sql-train  | 
rl-text2sql-train  | Question:
rl-text2sql-train  | Return all the committees that have delegates from Democratic party.
rl-text2sql-train  | 
rl-text2sql-train  | Generate ONLY the SQL query.
rl-text2sql-train  | Do not explain anything.
rl-text2sql-train  | 
rl-text2sql-train  | [GOLD SQL]
rl-text2sql-train  | SELECT T1.Committee FROM election AS T1 JOIN party AS T2 ON T1.Party  =  T2.Party_ID WHERE T2.Party  =  "Democratic"
rl-text2sql-train  | 
rl-text2sql-train  | [GRPO Mean KL]: 0.000000
rl-text2sql-train  | 
rl-text2sql-train  | --- Sample 0 | Reward: 5.5000 | KL: 0.0000 ---
rl-text2sql-train  | [GENERATED SQL]
rl-text2sql-train  | SELECT T3.Committee 
rl-text2sql-train  | FROM election AS T1 
rl-text2sql-train  | JOIN party AS T2 ON T1.Party = T2.Party_ID 
rl-text2sql-train  | JOIN county AS T4 ON T1.District = T4.County_Id 
rl-text2sql-train  | JOIN election AS T3 ON T2.Party_ID = T3.Party 
rl-text2sql-train  | WHERE T2.Party = 'Democratic' AND T3.Delegate IS NOT NULL;
rl-text2sql-train  | [RESULTS]
rl-text2sql-train  |   Rows Returned: 10
rl-text2sql-train  |   Top 3 Rows: [('Appropriations',), ('Appropriations',), ('Economic Matters',)]
rl-text2sql-train  | 
rl-text2sql-train  | [GOLD RESULTS]
rl-text2sql-train  |   Rows Returned: 4
rl-text2sql-train  |   Top 3 Rows: [('Appropriations',), ('Economic Matters',), ('Environmental Matters',)]
rl-text2sql-train  | 
rl-text2sql-train  | --- Sample 1 | Reward: -0.5000 | KL: 0.0000 ---
rl-text2sql-train  | [GENERATED SQL]
rl-text2sql-train  | SELECT T3.Committee 
rl-text2sql-train  | FROM election AS T1 
rl-text2sql-train  | JOIN party AS T2 ON T1.Party = T2.`Party_ID` 
rl-text2sql-train  | JOIN county AS T4 ON T1.District = T4.County_Id 
rl-text2sql-train  | WHERE T2.`Party` = 'Democratic' 
rl-text2sql-train  | GROUP BY T3.Committee;
rl-text2sql-train  | [EXECUTION ERROR] no such column: T3.Committee
rl-text2sql-train  | ==================================================
rl-text2sql-train  | 
rl-text2sql-train  | Train Avg Reward: 2.6391988039016723
rl-text2sql-train  | Average Loss: 0.000832366943359375
rl-text2sql-train  | Validation Avg Reward: 0.2
rl-text2sql-train exited with code 0
```
