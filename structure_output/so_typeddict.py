from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Annotated

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

#structured output with TypedDict
class Review(TypedDict):
    sentiment: Annotated[str, "The overall sentiment of the review, e.g., positive, negative, neutral"]
    pros: list[str]
    cons: list[str]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("The phone performs well overall, but the price is too high for the improvements offered. Battery life is decent, but charging is still slower than many Android phones.")

print(result)
# how to extract the sentiment from the structured output
sentiment = result["sentiment"]
print(f"Sentiment: {sentiment}")


# in typeddict there is no data validation , to do so we have to use pydantic model instead of typeddict, but the output will be similar to typeddict, and we can use the pydantic model to validate the output from the model.

# behind the scenes, the model is still generating text, but it is being parsed into a structured format based on the TypedDict definition. This allows for more predictable and usable outputs from the model.