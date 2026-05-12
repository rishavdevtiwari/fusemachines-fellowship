from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from crud import orderdetail_crud
from database import get_db

router = APIRouter(prefix="/orderdetails", tags=["OrderDetails"])

@router.get("/count")
def get_orderdetails_count(db: Session = Depends(get_db)):
    return {"table": "orderdetails", "count": orderdetail_crud.get_orderdetails_count(db)}

@router.get("/")
def get_orderdetails(db: Session = Depends(get_db)):
    return orderdetail_crud.get_all_orderdetails(db)