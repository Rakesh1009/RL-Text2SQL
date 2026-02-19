import os
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

DRIVE_MODEL_ROOT = "/content/drive/MyDrive/RL_Text2SQL_storage/models"


def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_model_path(model_name):
    local_dir = os.path.join(
        DRIVE_MODEL_ROOT,
        model_name.replace("/", "_")
    )

    if os.path.exists(local_dir):
        print(f"Loading model from local cache: {local_dir}")
        return local_dir

    print("Model not found locally. Downloading from HuggingFace...")
    return model_name


def load_model(config_path="configs/default.yaml"):
    config = load_config(config_path)
    model_cfg = config["model"]

    model_name = model_cfg["name"]
    resolved_path = resolve_model_path(model_name)

    print(f"Using model: {resolved_path}")

    # 4bit QLoRA config
    if model_cfg.get("load_in_4bit", True):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_path,
        cache_dir=DRIVE_MODEL_ROOT
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=DRIVE_MODEL_ROOT
    )

    # Save to Drive if downloaded
    if resolved_path == model_name:
        save_path = os.path.join(
            DRIVE_MODEL_ROOT,
            model_name.replace("/", "_")
        )
        print(f"Saving model locally to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

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
