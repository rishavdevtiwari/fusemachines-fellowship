from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from crud import office_crud
from database import get_db
from logger import logger

router = APIRouter(prefix="/offices", tags=["Offices"])

@router.get("/count")
def get_offices_count(db: Session = Depends(get_db)):
    return {"table": "offices", "count": office_crud.get_offices_count(db)}

@router.get("/")
def get_offices(db: Session = Depends(get_db)):
    return office_crud.get_all_offices(db)