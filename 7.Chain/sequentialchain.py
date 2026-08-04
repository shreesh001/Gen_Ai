from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

prompt1 = PromptTemplate(
    template='Provide a comprehensive, scene-by-scene summary of the movie {movie_name}.',
    input_variables=['movie_name']
)

prompt2 = PromptTemplate(
    template='Write a full short summary of the following movie {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

chain.get_graph().print_ascii()  
result = chain.invoke('Inception')

print(result)


