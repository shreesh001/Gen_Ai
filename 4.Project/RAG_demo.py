from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar is called the God of Cricket.",
    "Rohit Sharma is famous for scoring three ODI double centuries.",
    "Jasprit Bumrah is India's premier fast bowler famous for yorkers."
]

query = "Tell me about Bumrah"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

top_k = np.argsort(scores)[::-1][:3]

context = "\n".join([documents[i] for i in top_k])

prompt = f"""
You are a cricket expert.

Context:
{context}

Question:
{query}

Answer only using the context.
"""

response = llm.invoke(prompt)

print(response.content)