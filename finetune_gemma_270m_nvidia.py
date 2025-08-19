from autotrain.params import LLMTrainingParams
from autotrain.project import AutoTrainProject

HF_USERNAME = ""  # your HF username
HF_TOKEN = ""     # your HF token

params = LLMTrainingParams(
    model="LiquidAI/LFM2-350M-GGUF",  # GGUF model
    data_path="argilla/distilabel-capybara-dpo-7k-binarized",
    prompt_text_column="prompt",
    text_column="chosen",          # the positive (accepted) responses
    rejected_text_column="rejected", # the negative responses
    epochs=1,
    batch_size=4,                  # higher batch size for NVIDIA GPU
    lr=2e-4,
    trainer="orpo",                # important: specify ORPO trainer
    chat_template="chatml",        # useful if your dataset is chat-style
    peft=True,                     # use parameter-efficient fine-tuning
    mixed_precision="bf16",        # use bfloat16 for NVIDIA GPU efficiency
    project_name="lfm2-350m-gguf-finetuned",
    push_to_hub=False,             # set True if you want to upload
    username=HF_USERNAME,
    token=HF_TOKEN,
    # GGUF model optimizations
    use_peft=True,                 # ensure PEFT is enabled
    quantization="none",           # GGUF models are already quantized
    lora_r=16,                     # LoRA rank
    lora_alpha=32,                 # LoRA alpha
    lora_dropout=0.1,             # LoRA dropout
    # Training optimizations for NVIDIA
    gradient_accumulation_steps=2, # lower for higher batch size
    warmup_ratio=0.1,              # warmup ratio
    weight_decay=0.01,             # weight decay
    max_grad_norm=1.0,             # gradient clipping
    # Authentication for model download
    use_auth_token=HF_TOKEN,       # explicitly pass token for model download
    # GGUF-specific settings
    use_flash_attention_2=False,   # GGUF models may not support Flash Attention 2
    use_gradient_checkpointing=True, # enable for memory efficiency
    use_bf16=True,                 # enable bfloat16 for NVIDIA
    use_fp16=False,                # prefer bfloat16 over fp16
    # Additional optimizations
    dataloader_num_workers=4,      # multiple workers for faster data loading
    save_strategy="epoch",         # save at end of each epoch
    evaluation_strategy="no",      # disable evaluation during training for simplicity
    # Model loading overrides for GGUF
    trust_remote_code=True,        # GGUF models may need this
    torch_dtype="auto",            # let torch decide the dtype
    device_map="auto",             # let accelerate handle device mapping
)

project = AutoTrainProject(
    params=params,
    backend="local",   # train locally
    process=True,      # whether to immediately start training
)

project.create()
