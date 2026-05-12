from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from crud import order_crud
from database import get_db

router = APIRouter(prefix="/orders", tags=["Orders"])

@router.get("/count")
def get_orders_count(db: Session = Depends(get_db)):
    return {"table": "orders", "count": order_crud.get_orders_count(db)}

@router.get("/")
def get_orders(db: Session = Depends(get_db)):
    return order_crud.get_all_orders(db)