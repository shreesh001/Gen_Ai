from langchain_community.document_loaders import TextLoader 
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

loader = TextLoader(file_path="sample.txt", encoding="utf-8")
documents = loader.load()

prompt = PromptTemplate(
    template='Answer the following question ->{question} \n from the given text ->{text}',
    input_variables=['question', 'text']
)

chain = prompt | chat_model

result = chain.invoke({'question':'What is a LLM?', 'text':documents[0].page_content})

print(result.content)