from kfp.dsl import component, Input, Output, Dataset, Model

@component(
    base_image="huggingface/transformers-pytorch-gpu:latest",
    packages_to_install=["datasets", "transformers"]
)
def fine_tune_llm(
    preprocessed_data: Input[Dataset],
    model_output: Output[Model],
    base_model: str = "deepseek-ai/DeepSeek-V3-0324",
    max_len: int = 512,
    epochs: int = 1,
    lr: float = 2e-4,
    batch_size: int = 1,
    grad_accum: int = 16,
):
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    ds = load_from_disk(preprocessed_data.path)

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def pack(ex):
        prompt = f"""
        ### Instruction: {ex['instruction']}\n
        ### Input: {ex['input']}\n
        ### Output: {ex['output']}
        """
        x = tok(prompt, truncation=True, padding="max_length", max_length=max_len)
        y = tok(ex["output"], truncation=True, padding="max_length", max_length=max_len)
        x["labels"] = y["input_ids"]
        return x
    
    tds = ds["train"].map(pack, remove_columns=ds["train"].column_names)
    vds = ds["test"].map(pack, remove_columns=ds["test"].column_names)

    model = AutoModelForCausalLM.from_pretrained(
        base_model, load_in_8bit=True, device_map="auto"
    )

    peft_cfg = LoraConfig(
        r=8, lora_alpha=32, target_modules=["q_proj","v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_cfg)

    args = TrainingArguments(
        output_dir=model_output.path,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=tds, eval_dataset=vds
    )
    trainer.train()

    trainer.save_model(model_output.path)
    tok.save_pretrained(model_output.path)