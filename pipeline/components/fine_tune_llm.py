from kfp.dsl import component, Input, Output, Dataset, Model

@component(
    base_image="huggingface/transformers-pytorch-gpu:latest",
    packages_to_install=["datasets", "transformers", "accelerate"]
)
def fine_tune_llm(
    preprocessed_data: Input[Dataset],
    model_output: Output[Model],
    base_model: str = "openai-community/gpt2",
    max_len: int = 512,
    epochs: int = 1,
    lr: float = 2e-4,
    batch_size: int = 1,
    grad_accum: int = 16,
):
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    print(f"path: {preprocessed_data.path}")

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
        base_model, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    args = TrainingArguments(
            output_dir=model_output.path,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=epochs,
            learning_rate=lr,
            logging_steps=10,
            logging_dir=f"{model_output}/logs",
            fp16=torch.cuda.is_available(),
            save_total_limit=1,
        )

    trainer = Trainer(
            model=model, 
            args=args, 
            train_dataset=tds, 
            eval_dataset=vds
        )
    trainer.train()

    trainer.save_model(model_output.path)
    tok.save_pretrained(model_output.path)

    # # Load preprocessed dataset
    # ds = load_from_disk(preprocessed_data.path)

    # # Load tokenizer
    # tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    # if tok.pad_token is None:
    #     tok.pad_token = tok.eos_token

    # # Prepare model
    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model,
    #     torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    #     device_map="auto" if torch.cuda.is_available() else None,
    # )

    # # For now just save the pretrained model
    # model.save_pretrained(model_output.path)
    # tok.save_pretrained(model_output.path)

    # print(f"✅ Model and tokenizer saved to {model_output.path}")