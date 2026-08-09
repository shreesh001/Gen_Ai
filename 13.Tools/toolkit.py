from langchain_core.tools import tool

@tool
def multiply(a,b):
    """multiply two numbers"""
    return a*b

@tool
def divide(a,b):
    """divide two numbers (when b!=0)"""
    return a/b

class MathToolkit:
    def get_tools(self):
        return [multiply,divide]


toolkit = MathToolkit()
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name, "=>", tool.description)