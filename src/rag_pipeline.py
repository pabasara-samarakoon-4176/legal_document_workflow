from dotenv import load_dotenv
import os
import json
from llama_index.core import Document, VectorStoreIndex, StorageContext
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document
from llama_index.core.text_splitter import TokenTextSplitter

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

CHROMA_DB_PATH = "./clause_chroma_db"

def load_clause_documents(path="/Users/pabasarasamarakoon/agent_document_workflow/legal_document_workflow/ProseObjects-01/clauses.jsonl"):
    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            documents.append(
                Document(
                    text=entry["text"],
                    metadata={
                        "title": entry.get("title", ""),
                        "section": entry.get("section", ""),
                        "id": entry.get("id", ""),
                        "source": entry.get("source", "")
                    }
                )
            )
    return documents

if not os.path.exists(CHROMA_DB_PATH):
    print("Creating new Chroma DB...")

    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
    nodes = text_splitter.get_nodes_from_documents(load_clause_documents())

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = chroma_client.get_or_create_collection("clauses")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = OpenAIEmbedding(model="text-embedding-3-small", batch_size=256)

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    print("Legal clauses indexed into ChromaDB ✅")
else:
    print("Chroma DB already exists. Loading existing data...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = chroma_client.get_or_create_collection("clauses")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    print("Existing index loaded.")

llm = OpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=5,
    response_mode="compact"
)

def search_clauses(query: str):
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(query)

    results = []
    for i, node in enumerate(nodes):
        results.append({
            "title": node.metadata.get("title", f"Clause {i+1}"),
            "section": node.metadata.get("section", ""),
            "text": node.text,
            "similarity": getattr(node, "score", 0.7),
            "source": node.metadata.get("source", "")
        })
    return results

# query = "termination of agreement"
# matches = search_clauses(query)

# for item in matches:
#     print(f"Clause: {item['title']} ({item['section']})")
#     print(f"Similarity: {item['similarity']}")
#     print(f"Text: {item['text'][:300]}...\n")