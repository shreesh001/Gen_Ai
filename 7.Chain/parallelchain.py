from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

prompt1 = PromptTemplate(
    template='give the very short summary of the following movie {movie_name}.',
    input_variables=['movie_name']
)

prompt2 = PromptTemplate(
    template='take a 5 question quiz on the following movie {movie_name}',
    input_variables=['movie_name']
)

prompt3= PromptTemplate(
    template='merge the following summary **{summary}** and quiz **{quiz}** into a single output',
    input_variables=['summary','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "summary": prompt1 | model | parser,
    "quiz": prompt2 | model | parser
})

chain = parallel_chain | prompt3 | model | parser

result = chain.invoke('Avengers: Endgame')

chain.get_graph().print_ascii()

print(result)
