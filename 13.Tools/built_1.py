from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

api_wrapper = DuckDuckGoSearchAPIWrapper(
    region="wt-wt",
    safesearch="moderate",
    time="y",
    max_results=5,
    backend="auto",
    source="text"
)

search = DuckDuckGoSearchRun(api_wrapper=api_wrapper)

result = search.invoke("Current weather of haldwani Uttarakhand")

print(result)

print(search.name)
print(search.description)
print(search.args)