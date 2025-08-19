import os
# Disable Flash Attention 2 for Mac MPS compatibility
os.environ["USE_FLASH_ATTENTION_2"] = "0"
os.environ["FLASH_ATTENTION_DISABLE"] = "1"
# Additional environment variables to disable problematic features
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from autotrain.params import LLMTrainingParams
from autotrain.project import AutoTrainProject

HF_USERNAME = ""  # your HF username
HF_TOKEN = ""     # your HF token - replace with your actual token

params = LLMTrainingParams(
    model=f"google/gemma-3-270m",  # Gemma models require authentication
    data_path="argilla/distilabel-capybara-dpo-7k-binarized",
    prompt_text_column="prompt",
    text_column="chosen",          # the positive (accepted) responses
    rejected_text_column="rejected", # the negative responses
    epochs=1,
    batch_size=1,
    lr=2e-4,
    trainer="orpo",                # important: specify ORPO trainer
    chat_template="chatml",        # useful if your dataset is chat-style
    peft=True,                     # use parameter-efficient fine-tuning
    mixed_precision="no",          # "bf16"/"fp16" won't work on Mac MPS
    project_name="gemma3-270m-finetuned",
    push_to_hub=False,             # set True if you want to upload
    username=HF_USERNAME,
    token=HF_TOKEN,
    # MPS-specific optimizations
    use_peft=True,                 # ensure PEFT is enabled
    quantization="none",           # disable quantization for MPS compatibility
    lora_r=16,                     # LoRA rank
    lora_alpha=32,                 # LoRA alpha
    lora_dropout=0.1,             # LoRA dropout
    # Disable problematic optimizations
    use_4bit=False,                # disable 4-bit quantization
    use_8bit=False,                # disable 8-bit quantization
    use_nested_quant=False,        # disable nested quantization
    # Training optimizations for MPS
    gradient_accumulation_steps=4, # reduce memory pressure
    warmup_ratio=0.1,              # warmup ratio
    weight_decay=0.01,             # weight decay
    max_grad_norm=1.0,             # gradient clipping
    # Authentication for Gemma model download
    use_auth_token=HF_TOKEN,       # explicitly pass token for model download
    # Disable unsupported features for Mac MPS
    use_flash_attention_2=False,   # disable Flash Attention 2 (not supported on MPS)
    use_gradient_checkpointing=False, # disable gradient checkpointing for MPS
    use_bf16=False,                # disable bfloat16 (not supported on MPS)
    use_fp16=False,                # disable fp16 (not supported on MPS)
    # Additional MPS compatibility
    dataloader_num_workers=0,      # reduce worker processes for MPS
    save_strategy="epoch",         # save at end of each epoch
    evaluation_strategy="no",      # disable evaluation during training for simplicity
    # Model loading overrides
    trust_remote_code=False,       # don't trust remote code
    torch_dtype="auto",            # let torch decide the dtype
    device_map="auto",             # let accelerate handle device mapping
)

project = AutoTrainProject(
    params=params,
    backend="local",   # train locally
    process=True,      # whether to immediately start training
)

project.create()
