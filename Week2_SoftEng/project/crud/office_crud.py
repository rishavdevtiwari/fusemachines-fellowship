from sqlalchemy.orm import Session
from sqlalchemy import text
from logger import logger

def get_offices_count(db: Session):
    logger.info("Executing query: SELECT COUNT(*) FROM offices")
    return db.execute(text("SELECT COUNT(*) FROM offices")).scalar()

def get_all_offices(db: Session):
    logger.info("Executing query: SELECT * FROM offices")
    return [dict(row) for row in db.execute(text("SELECT * FROM offices")).mappings().all()]