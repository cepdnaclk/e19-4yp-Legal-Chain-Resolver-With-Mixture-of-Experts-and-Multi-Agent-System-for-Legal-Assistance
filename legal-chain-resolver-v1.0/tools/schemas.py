from pydantic import BaseModel

class TravelInfo(BaseModel):
    origin: str
    destination: str
    start_date: str
    end_date: str
    budget_amount: float
    budget_currency: str