
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator

class ModelInfo(BaseModel):
    """Information about a model provided by an API."""
    
    str_identifier: str = Field(..., description="String that identifies this model in the provider API")
    price_in: float = Field(..., description="Price per 1 million input tokens. 0 if free")
    price_out: float = Field(..., description="Price per 1 million output tokens. 0 if free")
    description: str = Field(..., description="Description of the model")
    creator: str = Field(..., description="Creator of the model (e.g., Google, OpenAI)")
    created: Optional[int] = Field(None, description="Unix timestamp (seconds since epoch) when the model was created")


    def __str__(self) -> str:
        """Return the string identifier of the model."""
        return self.str_identifier
    
    def __repr__(self) -> str:
        """Return a detailed representation of the model."""
        return f"ModelInfo(id={self.str_identifier}, creator={self.creator})"
    
    def get_creation_date(self) -> Optional[datetime]:
        """Convert the Unix timestamp to a datetime object if available."""
        if self.created is not None:
            return datetime.fromtimestamp(self.created)
        return None