from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='write a small poem on the following topic {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='explain the following joke \n {text}',
    input_variables=['text']
)

chain = prompt1 | model | parser

parallel_chain = RunnableParallel(
    {
        'Joke':RunnablePassthrough(),
        'Explaination':RunnableSequence(prompt2,model,parser)
    }
)

final_chain = chain | parallel_chain

res = final_chain.invoke({'topic':'AI'})

print(res['Joke'])
print(res['Explaination'])
