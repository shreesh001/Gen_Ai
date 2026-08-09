from langchain_core.tools import tool

# steps to create a custom tool
# Step 1 - create a function
# Step 2 - add type hints and description
# Step 3 - add tool decorator
@tool
def multipy(a:int, b:int) ->int:
    """multipy two numbers"""
    return a*b

res = multipy.invoke({"a":1, "b":3})

print(multipy.name,"\n")
print(multipy.description,"\n")
print(multipy.args,"\n")

