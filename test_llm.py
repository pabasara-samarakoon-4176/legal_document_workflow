import os
from typing import List, Optional
from dotenv import load_dotenv

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

load_dotenv()
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables")

llm = HuggingFaceInferenceAPI(
    model_name="deepseek-ai/DeepSeek-V3-0324",
    token=HF_TOKEN,
    provider="auto", 
)

response = llm.complete("What is the capital of France?")
print(response)