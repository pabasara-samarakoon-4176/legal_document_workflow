from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = "cpu"

app = FastAPI()

model_dir = os.environ.get("AIP_STORAGE_URI", "/model")
model_dir = "/Users/pabasarasamarakoon/agent_document_workflow/legal_document_workflow/pipeline/model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)  # CPU

class Request(BaseModel):
    prompt: str
    max_new_tokens: int = 64

@app.post("/predict")
def predict(req: Request):
    output = generator(req.prompt, max_new_tokens=req.max_new_tokens, do_sample=True)
    return {"predictions": [output[0]["generated_text"]]}

@app.get("/")
def health_check():
    return {"status": "healthy"}