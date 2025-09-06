import os
import re
import json
from pathlib import Path

# --- Helpers ---
def normalize_placeholders(text: str) -> str:
    """
    Replace CommonAccord-style placeholders like {_Customer} with 'Customer'
    """
    return re.sub(r"{_([A-Za-z0-9]+)}", r"\1", text)

def parse_md_file(file_path: Path, section_name: str):
    """
    Parse a single CommonAccord .md clause file and return a list of clauses
    """
    clauses = []
    current_id = None
    current_text = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("="):  # skip formatting like =[...]
                continue

            # Title line: Ti=Confidentiality
            if line.startswith("Ti="):
                title = normalize_placeholders(line.split("=", 1)[1].strip())
                continue

            # Clause line like 1.1.sec=...
            match = re.match(r"([\d\.]+)\.sec=(.*)", line)
            if match:
                # save previous clause if exists
                if current_id and current_text:
                    clauses.append({
                        "id": current_id,
                        "text": normalize_placeholders(" ".join(current_text))
                    })
                current_id = match.group(1)
                current_text = [match.group(2).strip()]
            else:
                # continuation of current clause
                if current_id:
                    current_text.append(line)

        # save the last clause
        if current_id and current_text:
            clauses.append({
                "id": current_id,
                "text": normalize_placeholders(" ".join(current_text))
            })

    return {
        "title": title if "title" in locals() else section_name,
        "section": section_name,
        "clauses": clauses,
        "source": str(file_path)
    }

# --- Main ---
def build_dataset(root_dir: str, output_file: str = "clauses.jsonl"):
    root_path = Path(root_dir)
    with open(output_file, "w", encoding="utf-8") as out:
        for section_dir in root_path.iterdir():
            if not section_dir.is_dir():
                continue
            section_name = section_dir.name
            for md_file in section_dir.glob("*.md"):
                parsed = parse_md_file(md_file, section_name)
                # write one clause per line (good for vector DB ingestion)
                for clause in parsed["clauses"]:
                    out.write(json.dumps({
                        "title": parsed["title"],
                        "section": parsed["section"],
                        "id": clause["id"],
                        "text": clause["text"],
                        "source": parsed["source"]
                    }) + "\n")

if __name__ == "__main__":
    # Example: point this to your ProseObjects-01/Sec directory
    build_dataset("Sec", "clauses.jsonl")
    print("✅ Dataset written to clauses.jsonl")
