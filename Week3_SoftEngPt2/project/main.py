from fastapi import FastAPI
import asyncio
from database import SessionLocal
from logger import logger

# 1. Import all routers
from routers import (
    product_router, customer_router, office_router, employee_router,
    order_router, orderdetail_router, payment_router, productline_router,
    agent_router,  # Task 4 deliverable: POST /agent/sql
)

# 2. Import all CRUD modules for the dashboard
from crud import (
    product_crud, customer_crud, office_crud, employee_crud,
    order_crud, orderdetail_crud, payment_crud, productline_crud,
)

app = FastAPI(
    title="ClassicModels API + Mini SQL Agent",
    description=(
        "Week 2 CRUD API for the ClassicModels database, plus the Week 3 "
        "Text-to-SQL agentic system (POST /agent/sql)."
    ),
    version="3.0.0",
)

# 3. Register all Routers
app.include_router(product_router.router)
app.include_router(customer_router.router)
app.include_router(office_router.router)
app.include_router(employee_router.router)
app.include_router(order_router.router)
app.include_router(orderdetail_router.router)
app.include_router(payment_router.router)
app.include_router(productline_router.router)
app.include_router(agent_router.router)

async def fetch_count(crud_function):
    """Wrapper to run synchronous SQLAlchemy queries concurrently with isolated sessions"""
    def sync_runner():
        db = SessionLocal() # Open a fresh, dedicated connection for this thread
        try:
            return crud_function(db) # Run the specific count query
        finally:
            db.close() # Safely close this specific connection
            
    return await asyncio.to_thread(sync_runner)

# 4. Removed the leftover 'db: Session = Depends(get_db)'
@app.get("/overall_counts", tags=["Dashboard"])
async def get_overall_counts(): 
    logger.info("Incoming request: GET /overall_counts")
    logger.info("Starting concurrent tasks...")
    
    # asyncio.gather runs all queries simultaneously
    results = await asyncio.gather(
        fetch_count(product_crud.get_products_count),
        fetch_count(customer_crud.get_customers_count),
        fetch_count(office_crud.get_offices_count),
        fetch_count(employee_crud.get_employees_count),
        fetch_count(order_crud.get_orders_count),
        fetch_count(orderdetail_crud.get_orderdetails_count),
        fetch_count(payment_crud.get_payments_count),
        fetch_count(productline_crud.get_productlines_count)
    )
    
    logger.info("All concurrent tasks completed successfully.")
    
    return {
        "products": results[0],
        "customers": results[1],
        "offices": results[2],
        "employees": results[3],
        "orders": results[4],
        "orderdetails": results[5],
        "payments": results[6],
        "productlines": results[7],
    }