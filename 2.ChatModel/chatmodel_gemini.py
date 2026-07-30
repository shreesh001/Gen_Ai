from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from dotenv import load_dotenv
import os

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

result = chat_model.invoke("What is the capital of France?")

print(result.content)