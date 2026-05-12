from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
import asyncio
from database import get_db
from logger import logger

# Import all routers
from routers import product_router, customer_router
# from routers import office_router, employee_router, order_router, orderdetail_router, payment_router, productline_router

# Import all CRUD modules for concurrency
from crud import product_crud, customer_crud
# from crud import office_crud, employee_crud, order_crud, orderdetail_crud, payment_crud, productline_crud

app = FastAPI(title="ClassicModels API", description="Week 2 Assignment API")

# Register all Routers
app.include_router(product_router.router)
app.include_router(customer_router.router)
# app.include_router(office_router.router)
# (Uncomment the remaining routers as you build them)

async def fetch_count(crud_function, db: Session):
    """Wrapper to run synchronous SQLAlchemy queries concurrently"""
    return await asyncio.to_thread(crud_function, db)

@app.get("/overall_counts", tags=["Dashboard"])
async def get_overall_counts(db: Session = Depends(get_db)):
    logger.info("Incoming request: GET /overall_counts")
    logger.info("Starting concurrent tasks...")
    
    # asyncio.gather runs all queries simultaneously
    results = await asyncio.gather(
        fetch_count(product_crud.get_products_count, db),
        fetch_count(customer_crud.get_customers_count, db),
        # fetch_count(office_crud.get_offices_count, db),
        # fetch_count(employee_crud.get_employees_count, db),
        # fetch_count(order_crud.get_orders_count, db),
        # fetch_count(orderdetail_crud.get_orderdetails_count, db),
        # fetch_count(payment_crud.get_payments_count, db),
        # fetch_count(productline_crud.get_productlines_count, db)
    )
    
    logger.info("All concurrent tasks completed successfully.")
    
    return {
        "products": results[0],
        "customers": results[1],
        # "offices": results[2],
        # "employees": results[3],
        # "orders": results[4],
        # "orderdetails": results[5],
        # "payments": results[6],
        # "productlines": results[7],
    }