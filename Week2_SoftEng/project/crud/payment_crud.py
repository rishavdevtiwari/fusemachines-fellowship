from sqlalchemy.orm import Session
from sqlalchemy import text
from logger import logger

def get_payments_count(db: Session):
    logger.info("Executing query: SELECT COUNT(*) FROM payments")
    return db.execute(text("SELECT COUNT(*) FROM payments")).scalar()

def get_all_payments(db: Session):
    logger.info("Executing query: SELECT * FROM payments")
    return [dict(row) for row in db.execute(text("SELECT * FROM payments")).mappings().all()]