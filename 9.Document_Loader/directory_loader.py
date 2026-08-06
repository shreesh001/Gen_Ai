from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
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

Directory_loader = DirectoryLoader(
    path="pdfs",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# more the number of pdfs in the directory, more the time it will take to load all the documents.
# also one page of pdf is considered as one document, so if a pdf has 10 pages, it will be considered as 10 documents.
# so instead of using load we use lazy_load which will load the documents one by one and we can iterate over them.

loaded_documents = Directory_loader.lazy_load()

prompt = PromptTemplate(
    template='Find out the error and inconsistency in the following resume ->{text}',
    input_variables=['text']    
)

chain = prompt | chat_model 

first_doc = next(loaded_documents)

result = chain.invoke({
    "text": first_doc.page_content
})

print(result.content)
  