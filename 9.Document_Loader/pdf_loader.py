from langchain_community.document_loaders import PyPDFLoader
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

loader = PyPDFLoader(file_path="resume.pdf")

loaded_documents = loader.load()

prompt = PromptTemplate(
    template='Summarize the following resume ->{text}',
    input_variables=['text']
)

chain = prompt | chat_model

print(loaded_documents[0].page_content)

result = chain.invoke({'text':loaded_documents[0].page_content})

print(result.content)
