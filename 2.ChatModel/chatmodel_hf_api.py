from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from pathlib import Path
from dotenv import load_dotenv
import os

env_path = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(dotenv_path=env_path)  # Load environment variables from .env file
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(
    llm=llm,
)   

result = model.invoke("what is the capital of France?")

print(result.content)
