from pydantic import BaseModel

class EmployeeCreate(BaseModel):
    employeeNumber: int
    lastName: str
    firstName: str
    extension: str
    email: str
    officeCode: str
    jobTitle: str

class EmployeeOut(EmployeeCreate):
    pass