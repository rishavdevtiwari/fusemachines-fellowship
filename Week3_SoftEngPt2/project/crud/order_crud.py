from sqlalchemy.orm import Session
from sqlalchemy import text
from logger import logger

def get_orders_count(db: Session):
    logger.info("Executing query: SELECT COUNT(*) FROM orders")
    return db.execute(text("SELECT COUNT(*) FROM orders")).scalar()

def get_all_orders(db: Session):
    logger.info("Executing query: SELECT * FROM orders")
    return [dict(row) for row in db.execute(text("SELECT * FROM orders")).mappings().all()]