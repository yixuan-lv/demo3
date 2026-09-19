import os

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(config):
    use_4bit = config.get("quantization_bit") == 4
    dtype = torch.bfloat16 if config.get("bf16", True) else torch.float16
    model_args = {
        "trust_remote_code": config.get("trust_remote_code", True),
        "torch_dtype": dtype,
    }

    if torch.cuda.is_available():
        model_args["device_map"] = {"": int(os.environ.get("LOCAL_RANK", 0))}
    if use_4bit:
        model_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("quantization_type", "nf4"),
            bnb_4bit_use_double_quant=config.get("double_quant", True),
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(config["model_name_or_path"], **model_args)
    if not torch.cuda.is_available():
        model.to("cpu")
    return model


def build_model(config):
    tokenizer = load_tokenizer(config["model_name_or_path"])
    model = load_base_model(config)
    use_4bit = config.get("quantization_bit") == 4

    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.get("gradient_checkpointing", True),
        )
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    rank = config.get("lora_rank", 16)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=config.get("lora_alpha", rank * 2),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("target_modules", TARGET_MODULES),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model, tokenizer


def load_adapter(config):
    tokenizer = load_tokenizer(config["model_name_or_path"])
    model = load_base_model(config)
    model = PeftModel.from_pretrained(model, config["adapter_name_or_path"])
    model.eval()
    return model, tokenizer
