import requests
from bs4 import BeautifulSoup
import os
import json
from dotenv import load_dotenv
from tqdm import tqdm
from llama_index.llms.openai import OpenAI

urls = [
    "https://www.sec.gov/Archives/edgar/data/797465/000143774922020170/ex_409970.htm",
    "https://www.sec.gov/Archives/edgar/data/868780/000086878025000009/ex1010dormanproducts-execu.htm",
    "https://www.sec.gov/Archives/edgar/data/868780/000156459021060184/dorm-ex102_6.htm",
    "https://www.sec.gov/Archives/edgar/data/836687/000083668725000019/f9Exhibith5dVIA-VIACNDA.htm",
    "https://www.sec.gov/Archives/edgar/data/1588972/000119312524064782/d747131dex99d3.htm",
    "https://www.sec.gov/Archives/edgar/data/1785345/000095017022004559/labp-ex10_9.htm",
    "https://www.sec.gov/Archives/edgar/data/1090012/000156459021023786/dvn-ex101_308.htm",
    "https://www.sec.gov/Archives/edgar/data/1090012/000095017022007007/dvn-ex10_1.htm",
    "https://www.sec.gov/Archives/edgar/data/1822928/000119312521243736/d171134d8k.htm",
    "https://www.sec.gov/Archives/edgar/data/1310114/000110465922077898/tm2220600d2_defa14a.htm",
    "https://www.sec.gov/Archives/edgar/data/1090012/000095017025065278/dvn-ex10_4.htm",
    "https://www.sec.gov/Archives/edgar/data/1606498/000160649824000094/avnsform8k_12032024ex101.htm",
    "https://www.sec.gov/Archives/edgar/data/24545/000110465923042784/tm2311706d2_ex10-1.htm",
    "https://www.sec.gov/Archives/edgar/data/1874178/000187417823000058/ex1022023q3.htm",
    "https://www.sec.gov/Archives/edgar/data/724742/000155837024012240/tphs-20240630xex10d2.htm",
    "https://www.sec.gov/Archives/edgar/data/1490978/000149097824000072/sdgr-20240630xex101friesne.htm",
    "https://www.sec.gov/Archives/edgar/data/1540615/000154061520000006/ex10.pdf",
    "https://www.sec.gov/Archives/edgar/data/1830210/000183021023000095/exhibit102crispconsultinga.htm",
    "https://www.sec.gov/Archives/edgar/data/1096296/000153949725002214/n4863exh10-1.htm",
    "https://www.sec.gov/Archives/edgar/data/1096296/000153949725002214/n4863exh10-6.htm",
    "https://www.sec.gov/Archives/edgar/data/1573221/000119312523246557/d505108dex102.htm",
    "https://www.sec.gov/Archives/edgar/data/1061027/000119312521007529/d82534dex1031.htm",
    "https://www.sec.gov/Archives/edgar/data/1690080/000121390025016103/ea0231758-8k_180life.htm",
    "https://www.sec.gov/Archives/edgar/data/1595893/000095017022002302/tptx-ex10_28.htm",
    "https://www.sec.gov/Archives/edgar/data/1872309/000119312521295416/d181580dex1013.htm",
    "https://www.sec.gov/Archives/edgar/data/1866820/000153949721000893/exh1.htm",
    "https://www.sec.gov/Archives/edgar/data/1832483/000121390023063662/ea180762ex10-14_patriciaacq.htm",
    "https://www.sec.gov/Archives/edgar/data/1872309/000119312521295416/d181580dex1015.htm",
    "https://www.sec.gov/Archives/edgar/data/1822829/000119312520295876/d29162dex1020.htm",
    "https://www.sec.gov/Archives/edgar/data/1657853/000165785322000076/exhibit1016amendedandresta.htm",
    "https://www.sec.gov/Archives/edgar/data/1657853/000165785324000105/ex1011-italianmasterleasea.htm",
    "https://www.sec.gov/Archives/edgar/data/1866894/000153949721000834/exhibit99-1.htm",
    "https://www.sec.gov/Archives/edgar/data/1913510/000110465924051842/vfs-20231231xex4d21.htm",
    "https://www.sec.gov/Archives/edgar/data/789933/000078993322000015/exhibit961-coteausk1300.htm",
    "https://www.sec.gov/Archives/edgar/data/1283699/000128369923000151/tmus09302023ex101.htm",
    "https://www.sec.gov/Archives/edgar/data/1884082/000119312522023680/d239003dex109.htm",
    "https://www.sec.gov/Archives/edgar/data/1884082/000188408224000008/ex4107-gee23x008.htm",
    "https://www.sec.gov/Archives/edgar/data/1884082/000188408224000008/ex4145-ps23x072restatedc.htm",
    "https://www.sec.gov/Archives/edgar/data/1884082/000188408224000008/ex4144-ps23x071restatedl.htm",
    "https://www.sec.gov/Archives/edgar/data/1884082/000119312522023680/d239003dex1011.htm",
    "https://www.sec.gov/Archives/edgar/data/1869974/000149315224051208/ex10-1.htm",
    "https://www.sec.gov/Archives/edgar/data/1401914/000140191424000038/exhibit101xoma-daretradi.htm",
    "https://www.sec.gov/Archives/edgar/data/1609809/000095017024126018/mcrb-ex10_6.htm",
    "https://www.sec.gov/Archives/edgar/data/1609809/000095017024126018/mcrb-ex10_7.htm",
    "https://www.sec.gov/Archives/edgar/data/1782999/000178299925000005/ex427rp-puretechroyaltypur.htm",
    "https://www.sec.gov/Archives/edgar/data/1968915/000162828023018828/exhibit104-form10.htm",
    "https://www.sec.gov/Archives/edgar/data/1697587/000147793222000142/insd_ex1014.htm",
    "https://www.sec.gov/Archives/edgar/data/866439/000168316822006305/darkpulse_ex9901.htm",
    "https://www.sec.gov/Archives/edgar/data/1422892/000121390022019821/ea158458ex10-1_singularity.htm",
    "https://www.sec.gov/Archives/edgar/data/1959023/000101376224001800/ea021015501ex10-1_safe.htm",
    "https://www.sec.gov/Archives/edgar/data/1591615/000165495424009447/vxit_ex991.htm",
    "https://www.sec.gov/Archives/edgar/data/1959023/000121390024101586/ea022231801ex10-1_safe.htm",
    "https://www.sec.gov/Archives/edgar/data/1386570/000162828022014881/chromadexcdxc-jointventure.htm",
    "https://www.sec.gov/Archives/edgar/data/1502377/000143774920023840/conta20200930_10q.htm",
    "https://www.sec.gov/Archives/edgar/data/1502377/000143774921002477/conta20201231_10q.htm",
    "https://www.sec.gov/Archives/edgar/data/1534155/000153415521000036/e14801-ex10_3.htm",
    "https://www.sec.gov/Archives/edgar/data/909037/000090903724000045/exhibit992.htm",
    "https://www.sec.gov/Archives/edgar/data/1329606/000149315223012437/ex10-136.htm",
    "https://www.sec.gov/Archives/edgar/data/1342423/000110465922112694/tm2229209d2_ex10-4.htm",
    "https://www.sec.gov/Archives/edgar/data/1671858/000095017025042320/spry-ex10_28.htm",
    "https://www.sec.gov/Archives/edgar/data/1000694/000100069422000004/nvax-20211231xex1037.htm",
    "https://www.sec.gov/Archives/edgar/data/1020214/000095017024026195/cers-ex10_6.htm",
    "https://www.sec.gov/Archives/edgar/data/1000694/000100069424000035/nvax-20240630xexhibit106.htm",
    "https://www.sec.gov/Archives/edgar/data/1528287/000162828021005481/exhibit1018-sx1.htm",
    "https://www.sec.gov/Archives/edgar/data/1340122/000134012223000113/clmt-20230930xex10d6.htm",
    "https://www.sec.gov/Archives/edgar/data/1034842/000155837024014862/rigl-20240930xex10d2.htm",
    "https://www.sec.gov/Archives/edgar/data/1635088/000114036122041175/brhc10044005_ex10-1.htm",
    "https://www.sec.gov/Archives/edgar/data/1520262/000095017023055178/alks-ex10_3.htm",
    "https://www.sec.gov/Archives/edgar/data/1811109/000160706222000754/ex2_2.htm",
    "https://www.sec.gov/Archives/edgar/data/1801198/000156459022012769/legn-ex47_396.htm",
    "https://www.sec.gov/Archives/edgar/data/1342287/000110465921054733/tm2113073d6_exd7.htm",
    "https://www.sec.gov/Archives/edgar/data/1618921/000161892123000062/a4q23exhibit1034.htm",
    "https://www.sec.gov/Archives/edgar/data/1618921/000119312523253805/d559360dex101.htm",
    "https://www.sec.gov/Archives/edgar/data/1095595/000095010320019978/dp138624_ex9901.htm",
    "https://www.sec.gov/Archives/edgar/data/1618921/000161892123000062/a4q23exhibit108.htm",
    "https://www.sec.gov/Archives/edgar/data/1618921/000119312522272556/d412292dex103.htm",
    "https://www.sec.gov/Archives/edgar/data/1618921/000161892122000004/a11302021exhibit105.htm",
    "https://www.sec.gov/Archives/edgar/data/1618921/000119312522272556/d412292dex102.htm",
    "https://www.sec.gov/Archives/edgar/data/1816815/000149315220018089/filename3.htm",
    "https://www.sec.gov/Archives/edgar/data/1816815/000149315220023056/ex10-7.htm",
    "https://www.sec.gov/Archives/edgar/data/1378125/000164033421000290/togl_ex1033.htm",
    "https://www.sec.gov/Archives/edgar/data/918965/000091896521000022/scansourceex10646302021.htm",
    "https://www.sec.gov/Archives/edgar/data/1697500/000155837025009911/sei-20250630xex10d4.htm",
    "https://www.sec.gov/Archives/edgar/data/1021917/000149315225004498/ex10-22.htm",
    "https://www.sec.gov/Archives/edgar/data/2009233/000121390025046757/ea023945301ex10-7_agrozinc.htm",
]

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

llm = OpenAI(model="gpt-4o-mini-2024-07-18", api_key=api_key)

def fetch_text_from_url(url: str) -> str:
    headers = {"User-Agent": "LegalDocScraper/1.0 (pabasara@example.com)"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    words = text.split()
    return " ".join(words[:50]) 

def safe_json_parse(text: str, fallback: dict) -> dict:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return fallback
    
# ------------------------------
# 1. Classification Dataset
# ------------------------------
classification_data = []
for url in tqdm(urls, desc="Building classification dataset"):
    doc_text = fetch_text_from_url(url)
    prompt = f"""
    You are a legal document classifier. Classify the following text:

    Document:
    \"\"\"{doc_text}\"\"\"

    Respond ONLY in JSON with:
    - document_type: type of legal document
    - legal_categories: list of legal domains
    - classification_confidence: float (0 to 1)
    """
    response = str(llm.complete(prompt))
    output_json = safe_json_parse(response, {
        "document_type": "Unknown",
        "legal_categories": [],
        "classification_confidence": 0.0
    })
    classification_data.append({
        "instruction": "Classify the following legal document.",
        "input": doc_text,
        "output": output_json
    })

with open("legal_classification_dataset_v2.jsonl", "w", encoding="utf-8") as f:
    for entry in classification_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ------------------------------
# 2. Clause Extraction Dataset
# ------------------------------
clause_data = []
for entry in tqdm(classification_data, desc="Building clause extraction dataset"):
    doc_text = entry["input"]
    prompt = f"""
    You are a legal clause extraction assistant.

    Task: Extract key clauses and identify missing ones.

    Document:
    \"\"\"{doc_text}\"\"\"

    Respond ONLY in JSON with:
    {{
      "clauses": {{
         "ClauseName": "Clause text..."
      }},
      "missing_clauses": ["..."]
    }}
    """
    response = str(llm.complete(prompt))
    output_json = safe_json_parse(response, {"clauses": {}, "missing_clauses": []})
    clause_data.append({
        "instruction": "Extract clauses and identify missing ones.",
        "input": doc_text,
        "output": output_json
    })

with open("legal_clause_extraction_dataset_v2.jsonl", "w", encoding="utf-8") as f:
    for entry in clause_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ------------------------------
# 3. Risk Assessment Dataset
# ------------------------------
risk_data = []
for entry in tqdm(clause_data, desc="Building risk assessment dataset"):
    doc_text = entry["input"]
    clauses = entry["output"].get("clauses", {})
    missing_clauses = entry["output"].get("missing_clauses", [])
    formatted_clauses = "\n".join([f"{k}: {v}" for k, v in clauses.items()]) or "None"
    context_blurb = f"Missing clauses: {', '.join(missing_clauses) if missing_clauses else 'None'}"

    prompt = f"""
    You are a legal compliance assistant.

    --- Clauses ---
    {formatted_clauses}

    --- Context ---
    {context_blurb}

    Please respond ONLY in JSON:
    - risk_scores: {{clause_name: float}}
    - flagged_issues: [ ... ]
    - compliance_summary: "..."
    """
    response = str(llm.complete(prompt))
    output_json = safe_json_parse(response, {
        "risk_scores": {},
        "flagged_issues": [],
        "compliance_summary": "Could not parse"
    })
    risk_data.append({
        "instruction": "Assess risks from clauses and context.",
        "input": f"Clauses: {formatted_clauses}\nContext: {context_blurb}",
        "output": output_json
    })

with open("risk_assessment_dataset_v2.jsonl", "w", encoding="utf-8") as f:
    for entry in risk_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("✅ All datasets created: legal_classification_dataset_v2.jsonl, legal_clause_extraction_dataset_v2.jsonl, risk_assessment_dataset_v2.jsonl")