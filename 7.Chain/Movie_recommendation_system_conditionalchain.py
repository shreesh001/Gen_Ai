from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field

# ---- setup ----
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

# ---- Step 1: structured output for ratings ----
class MovieRating(BaseModel):
    imdb_rating: float = Field(description="IMDB rating converted to a scale of 100")
    rotten_tomatoes_rating: float = Field(description="Rotten Tomatoes rating converted to a scale of 100")
    average_rating: float = Field(description="Average of imdb_rating and rotten_tomatoes_rating")

parser2 = PydanticOutputParser(pydantic_object=MovieRating)

prompt1 = PromptTemplate(
    template=(
        "Find the IMDB rating and Rotten Tomatoes rating of the movie '{movie}'.\n"
        "Convert both ratings to a scale of 100 (same scale, same type).\n"
        "Then calculate their average.\n"
        "{format_instruction}"
    ),
    input_variables=['movie'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

rating_chain = prompt1 | model | parser2

# ---- Step 2: prompts for each branch ----
prompt2 = PromptTemplate(
    template=(
        "The average rating of the movie '{movie}' is {average_rating}/100.\n"
        "This is a good score. Write a short recommendation telling the user to WATCH this movie."
    ),
    input_variables=['movie', 'average_rating']
)

prompt3 = PromptTemplate(
    template=(
        "The average rating of the movie '{movie}' is {average_rating}/100.\n"
        "This is a low score. Write a short recommendation telling the user to NOT WATCH this movie."
    ),
    input_variables=['movie', 'average_rating']
)

# ---- Step 3: conditional branch ----
branch_chain = RunnableBranch(
    (lambda x: x.average_rating >= 60, RunnableLambda(lambda x: {"movie": movie_name, "average_rating": x.average_rating}) | prompt2 | model | parser),
    (lambda x: x.average_rating < 60, RunnableLambda(lambda x: {"movie": movie_name, "average_rating": x.average_rating}) | prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not determine rating")
)

# ---- Final chain ----
chain = rating_chain | branch_chain

movie_name = "Aashiqui 2"   # movie name yaha se pick ho raha hai branch ke andar
print(chain.invoke({'movie': movie_name}))

chain.get_graph().print_ascii()