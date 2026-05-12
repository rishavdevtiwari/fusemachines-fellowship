from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from crud import customer_crud
from database import get_db
from logger import logger

router = APIRouter(prefix="/customers", tags=["Customers"])

@router.get("/count")
def get_customers_count(db: Session = Depends(get_db)):
    logger.info("Incoming request: GET /customers/count")
    count = customer_crud.get_customers_count(db)
    return {"table": "customers", "count": count}

@router.get("/")
def get_customers(db: Session = Depends(get_db)):
    logger.info("Incoming request: GET /customers")
    return customer_crud.get_all_customers(db)