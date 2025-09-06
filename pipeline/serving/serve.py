from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

app = FastAPI()
model_dir = os.environ.get("MODEL_DIR","/model")
tok = AutoTokenizer.from_pretrained(model_dir)
mdl = AutoModelForCausalLM.from_pretrained(model_dir)
gen = pipeline("text-generation", model=mdl, tokenizer=tok, device=0)

class Req(BaseModel):
    prompt: str
    max_new_tokens: int = 128

@app.post("/predict")
def predict(r: Req):
    out = gen(r.prompt, max_new_tokens=r.max_new_tokens, do_sample=False)[0]["generated_text"]
    return {"predictions":[out]}