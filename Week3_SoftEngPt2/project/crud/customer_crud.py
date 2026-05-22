from sqlalchemy.orm import Session
from sqlalchemy import text
from logger import logger

def get_customers_count(db: Session):
    logger.info("Executing query: SELECT COUNT(*) FROM customers")
    result = db.execute(text("SELECT COUNT(*) FROM customers")).scalar()
    return result

def get_all_customers(db: Session):
    logger.info("Executing query: SELECT * FROM customers")
    result = db.execute(text("SELECT * FROM customers")).mappings().all()
    return [dict(row) for row in result]