from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pathlib import Path
from dotenv import load_dotenv
import os

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

os.environ["USER_AGENT"] = "GenAI Learning"

loader = WebBaseLoader(
    web_paths=("https://en.wikipedia.org/wiki/Artificial_intelligence",)
)

docs = loader.load()

print(docs[0].page_content[:1000])
