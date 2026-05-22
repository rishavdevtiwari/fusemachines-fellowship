from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from crud import productline_crud
from database import get_db
from logger import logger

router = APIRouter(prefix="/productlines", tags=["ProductLines"])

@router.get("/count")
def get_productlines_count(db: Session = Depends(get_db)):
    logger.info("Incoming request: GET /productlines/count")
    return {"table": "productlines", "count": productline_crud.get_productlines_count(db)}

@router.get("/")
def get_productlines(db: Session = Depends(get_db)):
    logger.info("Incoming request: GET /productlines")
    return productline_crud.get_all_productlines(db)