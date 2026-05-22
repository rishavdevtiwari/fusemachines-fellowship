from pydantic import BaseModel
from typing import Optional

class ProductLineCreate(BaseModel):
    productLine: str
    textDescription: str

class ProductLineOut(ProductLineCreate):
    pass