from kfp.dsl import component, Output, Dataset
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

@component(
    base_image="huggingface/transformers-pytorch-gpu:latest",
    packages_to_install=["datasets", "gcsfs"]
)
def preprocess_jsonl(
    jsonl_path: str, 
    preprocessed_output: Output[Dataset]
    ):
    import json, gcsfs
    from datasets import Dataset, DatasetDict

    fs = gcsfs.GCSFileSystem()
    standardized_data = []

    with fs.open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if 'output' in entry:
                if isinstance(entry['output'], (list, dict)):
                    entry['output'] = json.dumps(entry['output'])
                else:
                    entry['output'] = str(entry['output'])
            standardized_data.append(entry)
        # Step 2: Create a Hugging Face Dataset from the standardized list
        if not standardized_data:
            print("Warning: No data was loaded from the JSONL file.")
            return
        
    ds = Dataset.from_list(standardized_data)
    
    # Step 3: Split the dataset and save it
    split = ds.train_test_split(test_size=0.1, seed=42)
    dd = DatasetDict({"train": split["train"], "test": split["test"]})
    dd.save_to_disk(preprocessed_output.path)

# @component(
#     base_image="huggingface/transformers-pytorch-gpu:latest",
#     packages_to_install=["datasets"]
# )
# def preprocess_jsonl(output_dir: Output[Dataset]):
#     from datasets import load_dataset, DatasetDict

#     dataset = load_dataset("Recognai/ag_news_corrected_labels", split="train")
#     dataset = dataset.remove_columns(["original_label"])
#     dataset = dataset.rename_column("corrected_label", "label")
#     label_map = {0: "world", 1: "business", 2: "sports", 3: "sci/tech"}
#     dataset = dataset.map(lambda x: {"label": label_map[x["label"]]})
#     dataset = dataset.shuffle(seed=42)
#     dataset = dataset.train_test_split(test_size=0.3, seed=42)
#     dataset = DatasetDict({
#         "train": dataset["train"],
#         "test": dataset["test"]
#     })
#     print(f"Saving preprocessed dataset to {output_dir.path}")
#     dataset.save_to_disk(output_dir.path)