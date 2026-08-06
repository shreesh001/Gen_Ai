from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a linkedin post on the following {topic}',
    input_variables=['topic']
) 

prompt2 = PromptTemplate(
    template='Write a twitter post on the following {topic}',
    input_variables=['topic']
)

branch_chain = RunnableParallel({
    "linkedin":RunnableSequence(prompt1,model,parser),
    "twitter":RunnableSequence(prompt2,model,parser)
})

res = branch_chain.invoke({'topic':'India'})

print(res['linkedin'])
print(res['twitter'])


