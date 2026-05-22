from pydantic import BaseModel

class OfficeCreate(BaseModel):
    officeCode: str
    city: str
    phone: str
    addressLine1: str
    country: str
    postalCode: str
    territory: str

class OfficeOut(OfficeCreate):
    pass