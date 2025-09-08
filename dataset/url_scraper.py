import requests
from bs4 import BeautifulSoup
import os
import json
from dotenv import load_dotenv
from typing import Optional
from tqdm import tqdm

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.openai import OpenAI


urls = [
    "https://www.sec.gov/Archives/edgar/data/868780/000156459022005966/dorm-ex1016_838.htm",
    "https://www.sec.gov/Archives/edgar/data/797465/000143774924019392/ex_684281.htm",
    "https://www.sec.gov/Archives/edgar/data/1604643/000160464323000046/vincentgrieco-evoquaxaremp.htm",
    "https://www.sec.gov/Archives/edgar/data/878726/000110465921005020/tm213516d1_ex10-1.htm",
    "https://www.sec.gov/Archives/edgar/data/1898496/000141057823000254/gety-20221231xex10d15.htm",
    "https://www.sec.gov/Archives/edgar/data/1409493/000119312524009272/d726079dex101.htm",
    "https://www.sec.gov/Archives/edgar/data/768899/000076889921000085/tbi10k122720ex1025.htm",
    "https://www.sec.gov/Archives/edgar/data/717538/000071753821000065/ex103-2021employmentagreem.htm",
    "https://www.sec.gov/Archives/edgar/data/861842/000143774920023058/ex_211981.htm",
    "https://www.sec.gov/Archives/edgar/data/1579214/000095017024053881/eex-ex10_43.htm"
]

load_dotenv()
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


llm = OpenAI(model="gpt-4o-mini-2024-07-18", api_key=api_key)
response = llm.complete("Say this is a test")
print(response)

# def fetch_text_from_url(url: str) -> str:
#     headers = {"User-Agent": "LegalDocScraper/1.0 (pabasara@example.com)"}
#     resp = requests.get(url, headers=headers)
#     resp.raise_for_status()

#     soup = BeautifulSoup(resp.text, "html.parser")
#     # get only visible text
#     for tag in soup(["script", "style", "table"]):
#         tag.decompose()
#     text = soup.get_text(separator=" ", strip=True)

#     words = text.split()
#     return " ".join(words[:500])  # first 500 words

# OUTPUT_FILE_1 = "clause_extraction_dataset.jsonl"
# OUTPUT_FILE_2 = "legal_classification_dataset.jsonl"

# dataset = []
# with open(INPUT_FILE, "r", encoding="utf-8") as f:
#     for line in f:
#         dataset.append(json.loads(line))

# output_dataset = []

# for entry in tqdm(dataset):
#     try:
#         doc_text = entry["input"]

#         extract_prompt = f"""
#         You are a legal clause extraction assistant.

#         Task: Given the following legal document text, extract key clauses and identify missing important clauses
#         such as termination, indemnity, or dispute resolution.

#         Document:
#         \"\"\"
#         {doc_text}
#         \"\"\"

#         Respond ONLY in JSON with the following format:
#         {{
#           "clauses": {{
#              "ClauseName": "Clause text..."
#           }},
#           "missing_clauses": ["..."]
#         }}
#         """

#         response = llm.complete(extract_prompt)

#         # Some LLMs return objects, ensure string
#         response_text = str(response)

#         # Clean accidental markdown fences or extra text
#         cleaned = (
#             response_text.replace("```json", "")
#                          .replace("```", "")
#                          .strip()
#         )

#         try:
#             output_json = json.loads(cleaned)
#         except Exception as e:
#             print(f"[WARN] Could not parse JSON: {e}\nRaw output:\n{response_text[:300]}")
#             output_json = {"clauses": {}, "missing_clauses": []}

#         output_dataset.append({
#             "instruction": "Extract clauses and identify missing ones.",
#             "input": doc_text,
#             "output": output_json
#         })

#     except Exception as e:
#         print(f"[ERROR] Failed: {e}")

# # ------------------------------
# # Save as JSONL
# # ------------------------------
# with open(OUTPUT_FILE_1, "w", encoding="utf-8") as f:
#     for entry in output_dataset:
#         f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# print(f"✅ Saved {len(output_dataset)} samples into {OUTPUT_FILE_1}")

