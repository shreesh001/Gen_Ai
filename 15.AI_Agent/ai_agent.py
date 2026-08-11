from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

search_tool = DuckDuckGoSearchRun()


@tool
def get_weather_data(city: str) -> str:
    """Fetches the current weather data for a given city"""
    url = f'http://api.weatherstack.com/current?access_key=da97dd59555a270a0163e3eb8ca66b0e&query={city}'
    response = requests.get(url)
    return str(response.json())


agent_executor = create_react_agent(llm, tools=[search_tool, get_weather_data])

response = agent_executor.invoke({
    "messages": [
        {"role": "user", "content": "Find the capital of Uttarakhand, then find it's current weather condition"}
    ]
})

print("\nFinal Answer:")
print(response["messages"][-1].content)