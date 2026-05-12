from pydantic import BaseModel
from datetime import datetime
from decimal import Decimal

class PaymentCreate(BaseModel):
    customerNumber: int
    checkNumber: str
    paymentDate: datetime
    amount: Decimal

class PaymentOut(PaymentCreate):
    pass