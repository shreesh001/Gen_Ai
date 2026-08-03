from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

#structured output with Pydantic model
class Review(BaseModel):
    sentiment: str = Field(..., description="The overall sentiment of the review, e.g., positive, negative, neutral")
    pros: list[str] = Field(..., description="List of pros mentioned in the review")
    cons: list[str] = Field(..., description="List of cons mentioned in the review")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("The phone performs well overall, but the price is too high for the improvements offered. Battery life is decent, but charging is still slower than many Android phones.")

print(f"Result Pydantic: {result}")    
print("\n")
# how to extract the sentiment from the structured output

result_dict = result.model_dump()
print(f"Result Dict: {result_dict}") 
print("\n")   

result_json = result.model_dump_json()
print(f"Result JSON: {result_json}")    
print("\n")

