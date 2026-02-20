import os
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Persistent Google Drive cache root
DRIVE_CACHE_ROOT = "/content/drive/MyDrive/RL_Text2SQL_storage/hf_cache"


def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


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
        DRIVE_CACHE_ROOT,
        f"{safe_name}_{suffix}"
    )

    os.makedirs(model_cache_dir, exist_ok=True)

    return model_cache_dir


def load_model(config_path="configs/default.yaml"):

    config = load_config(config_path)
    model_cfg = config["model"]

    model_name = model_cfg["name"]
    use_4bit = model_cfg.get("load_in_4bit", True)

    print(f"\nLoading model: {model_name}")
    print(f"4-bit quantization enabled: {use_4bit}")

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
    # Apply LoRA
    # --------------------------------------------------
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

    return model, tokenizer