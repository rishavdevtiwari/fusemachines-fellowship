from pydantic import BaseModel
from typing import Optional

class CustomerCreate(BaseModel):
    customerNumber: int
    customerName: str
    contactLastName: str
    contactFirstName: str
    phone: str
    addressLine1: str
    city: str
    country: str

class CustomerOut(CustomerCreate):
    pass