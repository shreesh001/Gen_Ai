#used to store the context for an llm model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

message =[
    SystemMessage("you are helpful ai assistent"),
    HumanMessage("My name is hero and i am 20 year old")
]

result = chat_model.invoke(message)

message.append(AIMessage(content=result.content))

query = HumanMessage("Can you tell my name and age?")
message.append(query)

output = chat_model.invoke(message)

print(output.content)


