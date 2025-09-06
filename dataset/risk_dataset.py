import requests
from bs4 import BeautifulSoup
import os
import json
from dotenv import load_dotenv
from typing import Optional
from tqdm import tqdm
from llama_index.llms.openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


llm = OpenAI(model="gpt-4o-mini-2024-07-18", api_key=api_key)

INPUT_FILE = "clause_extraction_dataset.jsonl"
OUTPUT_FILE = "risk_assessment_dataset.jsonl"

dataset = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))

output_dataset = []

for entry in tqdm(dataset):
    doc_text = entry["input"]
    clauses = entry["output"].get("clauses", {})
    missing_clauses = entry["output"].get("missing_clauses", [])

    # Format clauses and context
    formatted_clauses = "\n".join([f"{k}: {v}" for k, v in clauses.items()]) or "None"
    context_blurb = f"Missing clauses: {', '.join(missing_clauses) if missing_clauses else 'None'}"

    prompt = f"""
    You are a legal compliance assistant.

    Below are clauses extracted from a legal document along with recommended improvements.
    Also included are reference clauses from a legal precedent knowledge base.

    --- Clauses ---
    {formatted_clauses}

    --- Context ---
    {context_blurb}

    Please:
    1. Identify any clauses that may pose legal risk or need revision.
    2. Assign a risk score between 0.0 (no risk) and 1.0 (high risk) for each clause.
    3. Flag missing or ambiguous sections.
    4. Provide a 1-sentence compliance summary.
    
    Respond the following in JSON format:
    - risk_scores: {{clause_name: float, ...}}
    - flagged_issues: [ ... ]
    - compliance_summary: "..."
    """
    response = llm.complete(prompt)
    response_text = str(response).strip()
    cleaned = response_text.replace("```json", "").replace("```", "").strip()
    try:
        output_json = json.loads(cleaned)
    except:
        output_json = {"risk_scores": {}, "flagged_issues": [], "compliance_summary": "Could not parse"}

    output_dataset.append({
        "instruction": "Assess risks from clauses and context.",
        "input": f"Clauses: {formatted_clauses}\nContext: {context_blurb}",
        "output": output_json
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in output_dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(output_dataset)} samples into {OUTPUT_FILE}")