from sqlalchemy.orm import Session
from sqlalchemy import text
from logger import logger

def get_orderdetails_count(db: Session):
    logger.info("Executing query: SELECT COUNT(*) FROM orderdetails")
    return db.execute(text("SELECT COUNT(*) FROM orderdetails")).scalar()

def get_all_orderdetails(db: Session):
    logger.info("Executing query: SELECT * FROM orderdetails")
    return [dict(row) for row in db.execute(text("SELECT * FROM orderdetails")).mappings().all()]