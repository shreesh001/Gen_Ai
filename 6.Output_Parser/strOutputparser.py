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

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a detail report on the following {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Write a five line summary on the following {text}',
    input_variables=['text']
)


#create a chain of prompts and models with output parsers
#string output parser is good to use with the chain of prompts and models, as it will return the output as a string, which can be used as input for the next prompt in the chain
chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke('Virat Kohli')

print(result)