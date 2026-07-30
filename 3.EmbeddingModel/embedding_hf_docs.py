from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

document =[
    "what is the capital of France?",
    "what is the capital of Germany?",
    "what is the capital of Italy?",
]

vector = embeddings.embed_documents(document)

print(len(vector))