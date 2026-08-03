from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

class MarvelCharacter(BaseModel):
    name: str = Field(..., description="The name of the Marvel Character")
    age: int = Field(..., description="The age of the Marvel Character")
    gender: str = Field(..., description="The gender of the Marvel Character")

parser = PydanticOutputParser(pydantic_object=MarvelCharacter)

prompt = PromptTemplate(
    template='Give me a name, age and gender of a any Marvel Character from the following country {country} \n {format_instructions}',
    input_variables=['country'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = prompt | model | parser

result = chain.invoke({'country': 'America'})

print(result)

