from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

#structured output with json
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "sentiment": {
      "type": "string",
      "description": "The overall sentiment of the review, e.g., positive, negative, neutral"
    },
    "pros": {
      "type": "array",
      "description": "List of pros mentioned in the review",
      "items": {
        "type": "string"
      }
    },
    "cons": {
      "type": "array",
      "description": "List of cons mentioned in the review",
      "items": {
        "type": "string"
      }
    }
  },
  "required": [
    "sentiment",
    "pros",
    "cons"
  ]
}

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("The phone performs well overall, but the price is too high for the improvements offered. Battery life is decent, but charging is still slower than many Android phones.")

print("\n")
print(f"Result in json format: {result}")    

# how to extract the sentiment from the structured output

