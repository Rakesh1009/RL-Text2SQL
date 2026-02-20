import os
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

# Persistent cache root (configurable via environment variable)
DRIVE_CACHE_ROOT = os.getenv("HF_CACHE_DIR", "./hf_cache")


def build_cache_dir(model_name: str, use_4bit: bool):
    """
    Build deterministic cache directory based on:
    - Model name
    - Quantization mode
    """

    safe_name = model_name.replace("/", "_")

    if use_4bit:
        suffix = "4bit"
    else:
        suffix = "fp16"

    model_cache_dir = os.path.join(
        os.getenv("HF_CACHE_DIR", "./hf_cache"),
        f"{safe_name}_{suffix}"
    )

    os.makedirs(model_cache_dir, exist_ok=True)

    return model_cache_dir


def load_model(config):
    model_cfg = config["model"]

    model_name = model_cfg["name"]
    use_4bit = model_cfg.get("load_in_4bit", True)
    
    rl_cfg = config.get("rl", {})
    use_kl = rl_cfg.get("use_kl", False)

    print(f"\nLoading model: {model_name}")
    print(f"4-bit quantization enabled: {use_4bit}")
    print(f"KL Regularization enabled: {use_kl}")

    # --------------------------------------------------
    # Build dedicated cache dir per quantization mode
    # --------------------------------------------------
    cache_dir = build_cache_dir(model_name, use_4bit)

    print(f"Using cache directory: {cache_dir}")

    # --------------------------------------------------
    # Quantization config
    # --------------------------------------------------
    quant_config = None

    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    # --------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.float16,
        cache_dir=cache_dir
    )

    # --------------------------------------------------
    # Load frozen reference model
    # --------------------------------------------------
    if use_kl:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            dtype=torch.float16,
            cache_dir=cache_dir
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    else:
        ref_model = None

    # --------------------------------------------------
    # Apply LoRA or Resume Checkpoint
    # --------------------------------------------------
    training_cfg = config.get("training", {})
    resume_path = training_cfg.get("resume", False)

    if isinstance(resume_path, str) and os.path.exists(resume_path):
        print(f"Resuming LoRA adapter from checkpoint: {resume_path}")
        model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=model_cfg["lora_r"],
            lora_alpha=model_cfg["lora_alpha"],
            target_modules=model_cfg["target_modules"],
            lora_dropout=model_cfg["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, ref_model, tokenizer