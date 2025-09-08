from kfp.dsl import component, Input, Dataset, Model

@component(
    base_image="huggingface/transformers-pytorch-gpu:latest",
    packages_to_install=["datasets","transformers","evaluate"]
)
def evaluate_llm(
    preprocessed_data: Input[Dataset],
    model_dir: Input[Model]
) -> float:
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import evaluate

    ds = load_from_disk(preprocessed_data.path)['test']
    tok = AutoTokenizer.from_pretrained(model_dir.path)
    mdl = AutoModelForCausalLM.from_pretrained(model_dir.path)
    gen = pipeline("text-generation", model=mdl, tokenizer=tok, device=0)

    rouge = evaluate.load("rouge")

    preds, refs = [], []
    sample = min(50, len(ds))
    for ex in ds.select(range(sample)):
        prompt = f"### Instruction: {ex['instruction']}\n### Input: {ex.get('input','')}\n### Output:"
        out = gen(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
        preds.append(out)
        refs.append(ex["output"])

    res = rouge.compute(predictions=preds, references=refs)
    print("Eval:", res)
    return float(res["rougeL"])
    # return 0.3