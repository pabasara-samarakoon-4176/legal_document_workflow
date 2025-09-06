from kfp.dsl import component, Output, Dataset

@component(
    base_image="huggingface/transformers-pytorch-gpu:latest",
    packages_to_install=["datasets"]
)
def preprocess_jsonl(jsonl_path: str, preprocessed_output: Output[Dataset]):
    from datasets import load_dataset
    ds = load_dataset("json", data_files={"train": jsonl_path})["train"]
    # 90/10 split
    split = ds.train_test_split(test_size=0.1, seed=42)
    from datasets import DatasetDict
    dd = DatasetDict({"train": split["train"], "test": split["test"]})
    dd.save_to_disk(preprocessed_output.path)