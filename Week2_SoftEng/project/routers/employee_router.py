from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from crud import employee_crud
from database import get_db

router = APIRouter(prefix="/employees", tags=["Employees"])

@router.get("/count")
def get_employees_count(db: Session = Depends(get_db)):
    return {"table": "employees", "count": employee_crud.get_employees_count(db)}

@router.get("/")
def get_employees(db: Session = Depends(get_db)):
    return employee_crud.get_all_employees(db)