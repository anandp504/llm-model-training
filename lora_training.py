from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOConfig, DPOTrainer
import os

HF_USERNAME = os.environ.get("HF_USERNAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def load_model():
    model_name = "LiquidAI/LFM2-350M"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading model (MPS-friendly) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": "mps"} if torch.backends.mps.is_available() else None,
        torch_dtype=torch.float32  # FP32 is safest on MPS
    )
    print("✅ Model loaded")
    print(f"Parameters: {model.num_parameters():,}")
    return model, tokenizer

def load_dpo_dataset():
    print("Loading dataset...")
    ds = load_dataset("mlabonne/orpo-dpo-mix-40k", split="train[:2000]")
    ds = ds.train_test_split(test_size=0.1, seed=42)
    print(f"Train samples: {len(ds['train'])}, Eval samples: {len(ds['test'])}")
    return ds["train"], ds["test"]

def wrap_model_peft(model):
    GLU_MODULES = ["w1", "w2", "w3"]
    MHA_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]
    CONV_MODULES = ["in_proj", "out_proj"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=GLU_MODULES + MHA_MODULES + CONV_MODULES,
        bias="none",
        modules_to_save=None,
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    print("✅ LoRA applied")
    return lora_model

def launch_training(lora_model, train_ds, eval_ds, tokenizer):
    dpo_config = DPOConfig(
        output_dir="./lfm2-dpo",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # accumulate to mimic larger batch
        learning_rate=1e-6,
        lr_scheduler_type="linear",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=False,     # disabled for macOS MPS compatibility
        fp16=False,     # macOS MPS is unstable with fp16
        optim="adamw_torch"  # no bitsandbytes
    )
    print("Initializing DPO trainer...")
    trainer = DPOTrainer(
        model=lora_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    print("Starting training...")
    trainer.train()
    trainer.save_model()
    print(f"✅ Model saved at: {dpo_config.output_dir}")

def save_merged_model(lora_model, tokenizer):
    print("Merging LoRA weights...")
    merged = lora_model.merge_and_unload()
    merged.save_pretrained("./lfm2-lora-merged")
    tokenizer.save_pretrained("./lfm2-lora-merged")
    print("✅ LoRA merged model saved at: ./lfm2-lora-merged")

def main():
    model, tokenizer = load_model()
    train_ds, eval_ds = load_dpo_dataset()
    lora_model = wrap_model_peft(model)
    launch_training(lora_model, train_ds, eval_ds, tokenizer)
    save_merged_model(lora_model, tokenizer)

if __name__ == "__main__":
    main()

