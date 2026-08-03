from pydantic import BaseModel, Field
from typing import Optional

class ReviewModel(BaseModel):
    sentiment: str = Field(..., description="The overall sentiment of the review, e.g., positive, negative, neutral")
    pros: list[str] = Field(..., description="List of pros mentioned in the review")
    cons: list[str] = Field(..., description="List of cons mentioned in the review") 
    Date: Optional[str] = Field(None, description="The date of the review, if available")  

new_review = ReviewModel(
    sentiment="positive",
    pros=["Good performance", "Decent battery life"],
    cons=["High price", "Slow charging"]     
    )

review_dict=dict(new_review)
review_json=new_review.model_dump_json()
print(review_json)
print(review_dict['sentiment'])