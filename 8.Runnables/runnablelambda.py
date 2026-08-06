from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

def countwords(text):
    return len(text.split())

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Tell me the definition of {topic}',
    input_variables=['topic']
)

chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'def':RunnablePassthrough(),
    'word_count':RunnableLambda(countwords)
})

final_chain = chain | parallel_chain

res = final_chain.invoke({'topic':'photosynthesis'})

print(res['def'])
print(res['word_count'])