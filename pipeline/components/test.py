import json
from datasets import load_dataset, DatasetDict, Dataset

jsonl_path = 'dataset/dataset_v1.jsonl'

standardized_data = []

with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f):
        try:
            entry = json.loads(line)
            
            # Ensure the 'output' column is a string
            if 'output' in entry:
                # If it's a list or dictionary, convert it to a JSON string
                if isinstance(entry['output'], (list, dict)):
                    entry['output'] = json.dumps(entry['output'])
                # If it's not a string, convert it to a string
                else:
                    entry['output'] = str(entry['output'])
            
            standardized_data.append(entry)
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {line_num + 1}: {e}")
            continue

ds = Dataset.from_list(standardized_data)
ds = DatasetDict({
    "train": ds.train_test_split(test_size=0.1, seed=42)["train"],
    "test": ds.train_test_split(test_size=0.1, seed=42)["test"]
})
ds.save_to_disk('dataset/preprocessed_data')