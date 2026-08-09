from langchain_community.tools import ShellTool

shell = ShellTool()

execute = shell.invoke("python built_1.py")


print(execute)

print(shell.name)
print(shell.description)
print(shell.args)