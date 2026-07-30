from langchain_openai import OpenAI 
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

llm = OpenAI(model_name="gpt-4", temperature=0.7, max_tokens=150)   

result = llm.invoke("Write a short poem about the beauty of nature.")

print(result)