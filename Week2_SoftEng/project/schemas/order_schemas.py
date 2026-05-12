from pydantic import BaseModel
from datetime import datetime

class OrderCreate(BaseModel):
    orderNumber: int
    orderDate: datetime
    requiredDate: datetime
    status: str
    customerNumber: int

class OrderOut(OrderCreate):
    pass