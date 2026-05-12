from sqlalchemy.orm import Session
from sqlalchemy import text
from logger import logger

def get_employees_count(db: Session):
    logger.info("Executing query: SELECT COUNT(*) FROM employees")
    return db.execute(text("SELECT COUNT(*) FROM employees")).scalar()

def get_all_employees(db: Session):
    logger.info("Executing query: SELECT * FROM employees")
    return [dict(row) for row in db.execute(text("SELECT * FROM employees")).mappings().all()]