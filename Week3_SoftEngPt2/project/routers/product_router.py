from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from crud import product_crud   # <-- Changed
from database import get_db
from logger import logger

router = APIRouter(prefix="/products", tags=["Products"]) # <-- Changed

@router.get("/count")
def get_products_count(db: Session = Depends(get_db)):
    logger.info("Incoming request: GET /products/count")
    return {"table": "products", "count": product_crud.get_products_count(db)}

@router.get("/")
def get_products(db: Session = Depends(get_db)):
    logger.info("Incoming request: GET /products")
    # Assuming you have a get_all_products in your crud file!
    return product_crud.get_all_products(db)