# this does not work because the Structured output parser is not compatible with new version of langchain 
# we have to use the pydantic model 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
# this below line cause the error 
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

schema =[
    ResponseSchema(name="name", description="The name of the Marvel Character"),    
    ResponseSchema(name="age", description="The age of the Marvel Character"),
    ResponseSchema(name="gender", description="The gender of the Marvel Character")
]

parser = StructuredOutputParser.from_response_schemas(schema)

prompt = PromptTemplate(
    template='Give me random name, age and gender of a any Marvel Character /n {format_instructions}',
    input_variables=[],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = prompt | model | parser

result = chain.invoke({})

print(result)