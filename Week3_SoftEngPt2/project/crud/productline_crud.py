from sqlalchemy.orm import Session
from sqlalchemy import text
from logger import logger

def get_productlines_count(db: Session):
    logger.info("Executing query: SELECT COUNT(*) FROM productlines")
    return db.execute(text("SELECT COUNT(*) FROM productlines")).scalar()

def get_all_productlines(db: Session):
    logger.info("Executing query: SELECT * FROM productlines")
    return [dict(row) for row in db.execute(text("SELECT * FROM productlines")).mappings().all()]