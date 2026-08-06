from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from dotenv import load_dotenv
import os

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

loader = CSVLoader(file_path="data.csv", encoding="utf-8")
documents = loader.load()
text = "\n\n".join(doc.page_content for doc in documents)

prompt = PromptTemplate(
    template="""
You are given employee data.

Data:
{text}

Question:
{question}

Answer only using the provided data.
""",
    input_variables=["text", "question"]
)
parser = StrOutputParser()

chain = prompt | chat_model | parser

questions = [
    "How many employees are there in the company?",
    "List the names of all employees working in the Software Engineering department.",
    "Who has the highest salary, and what are their designation and department?",
    "Find all employees who have more than 5 years of experience and a performance rating of at least 4.5.",
    "Based on salary, experience, and performance rating, recommend the top 3 employees who are most deserving of a promotion and explain why."
]

for query in questions:
    result = chain.invoke({'question': query, 'text': text})
    print(f"Question: {query}")
    print(f"Answer: {result}\n")    
