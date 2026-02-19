import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import yaml


def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(config_path="configs/default.yaml"):
    config = load_config(config_path)
    model_cfg = config["model"]

    model_name = model_cfg["name"]

    print(f"Loading model: {model_name}")

    # QLoRA config
    if model_cfg.get("load_in_4bit", True):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

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
