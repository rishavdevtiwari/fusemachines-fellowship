from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from crud import payment_crud
from database import get_db

router = APIRouter(prefix="/payments", tags=["Payments"])

@router.get("/count")
def get_payments_count(db: Session = Depends(get_db)):
    return {"table": "payments", "count": payment_crud.get_payments_count(db)}

@router.get("/")
def get_payments(db: Session = Depends(get_db)):
    return payment_crud.get_all_payments(db)