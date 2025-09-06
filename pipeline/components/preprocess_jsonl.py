from kfp.dsl import component, Output, Dataset

@component(
    base_image="huggingface/transformers-pytorch-gpu:latest",
    packages_to_install=["datasets", "gcsfs"]
)
def preprocess_jsonl(jsonl_path: str, preprocessed_output: Output[Dataset]):
    import json
    from datasets import Dataset, DatasetDict

    # Step 1: Read and sanitize the JSONL file line by line
    standardized_data = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    entry = json.loads(line)
                    # Standardize the 'output' field to be a string.
                    if 'output' in entry:
                        if isinstance(entry['output'], (list, dict)):
                            entry['output'] = json.dumps(entry['output'])
                        else:
                            entry['output'] = str(entry['output'])
                    standardized_data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {line_num + 1}: {e}")
                    # Skip the problematic line to prevent pipeline failure
                    continue

    except FileNotFoundError:
        print(f"Error: The file at {jsonl_path} was not found.")
        return

    # Step 2: Create a Hugging Face Dataset from the standardized list
    if not standardized_data:
        print("Warning: No data was loaded from the JSONL file.")
        return
        
    ds = Dataset.from_list(standardized_data)
    
    # Step 3: Split the dataset and save it
    split = ds.train_test_split(test_size=0.1, seed=42)
    dd = DatasetDict({"train": split["train"], "test": split["test"]})
    dd.save_to_disk(preprocessed_output.path)