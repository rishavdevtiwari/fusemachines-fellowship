from pydantic import BaseModel
from decimal import Decimal

class OrderDetailCreate(BaseModel):
    orderNumber: int
    productCode: str
    quantityOrdered: int
    priceEach: Decimal
    orderLineNumber: int

class OrderDetailOut(OrderDetailCreate):
    pass