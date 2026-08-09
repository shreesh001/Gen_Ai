from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
)

loader = PyPDFLoader(file_path="OOP.pdf")

loaded_documents = loader.load()

split_documents = splitter.split_documents(loaded_documents)

print(split_documents[0].page_content,"\n\n",split_documents[1].page_content)

