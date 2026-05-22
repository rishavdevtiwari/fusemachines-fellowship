# Task 2 — Query Understanding (Structured Decomposition)

**Author:** Rishav Dev Tiwari
**Goal:** Break each natural-language question into structured components *before* writing SQL.

This document is the human-readable companion to `Task2_Decomposition.json`. The JSON is consumed directly by the Task 3 pipeline (`text2sql/sql_generator.py`) to render SQL automatically.

---

## Decomposition schema

For every question I capture six fields (the pillars of any SQL query):

| Field | Meaning |
|-------|---------|
| **Intent** | One-line description of what the question is asking for |
| **Tables** | Which tables are involved (with aliases for joins) |
| **Columns** | Which columns to project (or aggregate expressions) |
| **Filters** | `WHERE` conditions (none of the 50 benchmark questions need filters; included for completeness) |
| **Joins** | Join type, table, and ON-condition |
| **Group By** | Columns to group by, if any |

I also record `Order By` and `Distinct` because they affect the surface query but flow naturally from the question wording (e.g. "vendor list" = distinct, "show MSRP" → highest first).

---

## Example (from the assignment)

**Question:** *How many customers are from the USA?*

| Field | Value |
|-------|-------|
| Intent | Count total customers in a country |
| Tables | `customers` |
| Columns | `COUNT(*) AS "totalCustomers"` |
| Filters | `country = 'USA'` |
| Joins | none |
| Group By | none |

This decomposition translates directly to:
```sql
SELECT COUNT(*) AS "totalCustomers"
FROM customers
WHERE "country" = 'USA';
```

---

## Section A — Single-table projections (Q1–Q20)

### Q1. List all products
- **Intent:** Retrieve every row from a single table
- **Tables:** `products`
- **Columns:** `*`
- **Filters:** none
- **Joins:** none
- **Group By:** none

### Q2. Get all customers
- **Intent:** Retrieve every row from a single table
- **Tables:** `customers`
- **Columns:** `*`
- **Filters:** none · **Joins:** none · **Group By:** none

### Q3. Show all orders
- **Intent:** Retrieve every row from a single table
- **Tables:** `orders` · **Columns:** `*`

### Q4. List all employees
- **Tables:** `employees` · **Columns:** `*`

### Q5. Get all offices
- **Tables:** `offices` · **Columns:** `*`

### Q6. Show all product lines
- **Tables:** `productlines` · **Columns:** `*`

### Q7. List all payments
- **Tables:** `payments` · **Columns:** `*`

### Q8. Get product names and prices
- **Intent:** Project name and both prices (MSRP = sell, buyPrice = cost)
- **Tables:** `products`
- **Columns:** `"productName"`, `"MSRP"`, `"buyPrice"`

### Q9. Get customer names and cities
- **Tables:** `customers` · **Columns:** `"customerName"`, `"city"`

### Q10. List employee first and last names
- **Tables:** `employees` · **Columns:** `"firstName"`, `"lastName"`

### Q11. Get all order dates
- **Intent:** Project each order with its date, sorted chronologically
- **Tables:** `orders`
- **Columns:** `"orderNumber"`, `"orderDate"`
- **Order By:** `"orderDate"`

### Q12. Show product vendor list
- **Intent:** Distinct vendors
- **Tables:** `products` · **Columns:** `"productVendor"`
- **Distinct:** true · **Order By:** `"productVendor"`

### Q13. Get all product codes
- **Tables:** `products` · **Columns:** `"productCode"` · **Order By:** `"productCode"`

### Q14. List all countries from offices
- **Intent:** Distinct list of countries · **Tables:** `offices` · **Columns:** `"country"`
- **Distinct:** true · **Order By:** `"country"`

### Q15. Show all order statuses
- **Tables:** `orders` · **Columns:** `"status"` · **Distinct:** true · **Order By:** `"status"`

### Q16. Get all payment amounts
- **Tables:** `payments` · **Columns:** `"amount"` · **Order By:** `"amount" DESC`

### Q17. List all job titles
- **Tables:** `employees` · **Columns:** `"jobTitle"` · **Distinct:** true · **Order By:** `"jobTitle"`

### Q18. Get customer phone numbers
- **Tables:** `customers` · **Columns:** `"customerName"`, `"phone"`

### Q19. Show product MSRP values
- **Tables:** `products` · **Columns:** `"productName"`, `"MSRP"` · **Order By:** `"MSRP" DESC`

### Q20. List order numbers
- **Tables:** `orders` · **Columns:** `"orderNumber"` · **Order By:** `"orderNumber"`

---

## Section B — Joins (Q21–Q30)

### Q21. Get orders with customer names
- **Intent:** Each order labelled with the buying customer's name
- **Tables:** `orders o`, `customers c`
- **Columns:** `o."orderNumber"`, `o."orderDate"`, `o."status"`, `c."customerName"`
- **Joins:** `INNER JOIN customers c ON c."customerNumber" = o."customerNumber"`
- **Order By:** `o."orderNumber"`

### Q22. Get employees with office city
- **Tables:** `employees e`, `offices o`
- **Columns:** `e."employeeNumber"`, `e."firstName"`, `e."lastName"`, `e."jobTitle"`, `o."city"`
- **Joins:** `INNER JOIN offices o ON o."officeCode" = e."officeCode"`

### Q23. Get payments with customer names
- **Tables:** `payments p`, `customers c`
- **Columns:** payment fields + `c."customerName"`
- **Joins:** `INNER JOIN customers c ON c."customerNumber" = p."customerNumber"`
- **Order By:** `p."paymentDate"`

### Q24. Get order details with product names
- **Tables:** `orderdetails od`, `products pr`
- **Columns:** orderdetails fields + `pr."productName"`
- **Joins:** `INNER JOIN products pr ON pr."productCode" = od."productCode"`

### Q25. Get products with product line description
- **Tables:** `products p`, `productlines pl`
- **Columns:** product fields + `pl."textDescription"`
- **Joins:** `INNER JOIN productlines pl ON pl."productLine" = p."productLine"`

### Q26. Get customers with sales rep names
- **Intent:** Match customers to their sales rep, including customers without one
- **Tables:** `customers c`, `employees e`
- **Columns:** `c."customerNumber"`, `c."customerName"`, `e."firstName" || ' ' || e."lastName" AS "salesRep"`
- **Joins:** `LEFT JOIN employees e ON e."employeeNumber" = c."salesRepEmployeeNumber"`
- **Why LEFT join:** some customers have no rep assigned

### Q27. Get orders with customer city
- **Tables:** `orders o`, `customers c` · join on `customerNumber`
- **Columns:** order fields + `c."customerName"`, `c."city"`

### Q28. Get employees and their manager
- **Intent:** Self-join on the `reportsTo` hierarchy
- **Tables:** `employees e` (employee), `employees m` (manager)
- **Columns:** `e.*` plus `m."firstName" || ' ' || m."lastName" AS "manager"`
- **Joins:** `LEFT JOIN employees m ON m."employeeNumber" = e."reportsTo"`
- **Why LEFT join:** the President's `reportsTo` is NULL

### Q29. Get orderdetails with product vendor
- **Tables:** `orderdetails od`, `products p`
- **Columns:** orderdetails fields + `p."productName"`, `p."productVendor"`
- **Joins:** `INNER JOIN products p ON p."productCode" = od."productCode"`

### Q30. Get payments with customer country
- **Tables:** `payments p`, `customers c` · join on `customerNumber`
- **Columns:** payment fields + `c."customerName"`, `c."country"`

---

## Section C — Aggregations with GROUP BY (Q31–Q40)

### Q31. Count customers per country
- **Intent:** Group customers by country and count
- **Tables:** `customers`
- **Columns:** `"country"`, `COUNT(*) AS "customerCount"`
- **Group By:** `"country"`
- **Order By:** `"customerCount" DESC`

### Q32. Total payments per customer
- **Tables:** `payments p`, `customers c`
- **Columns:** `p."customerNumber"`, `c."customerName"`, `SUM(p."amount") AS "totalPaid"`
- **Joins:** `INNER JOIN customers c ON c."customerNumber" = p."customerNumber"`
- **Group By:** `p."customerNumber"`, `c."customerName"`

### Q33. Number of orders per status
- **Tables:** `orders` · **Columns:** `"status"`, `COUNT(*) AS "orderCount"` · **Group By:** `"status"`

### Q34. Products per product line
- **Tables:** `products` · **Columns:** `"productLine"`, `COUNT(*) AS "productCount"` · **Group By:** `"productLine"`

### Q35. Employees per office
- **Tables:** `offices o`, `employees e`
- **Columns:** `o."officeCode"`, `o."city"`, `COUNT(e."employeeNumber") AS "employeeCount"`
- **Joins:** `LEFT JOIN employees e ON e."officeCode" = o."officeCode"`
- **Group By:** `o."officeCode"`, `o."city"`

### Q36. Total stock per product vendor
- **Tables:** `products` · **Columns:** `"productVendor"`, `SUM("quantityInStock") AS "totalStock"`
- **Group By:** `"productVendor"`

### Q37. Average buy price per product line
- **Tables:** `products`
- **Columns:** `"productLine"`, `ROUND(AVG("buyPrice"), 2) AS "avgBuyPrice"`
- **Group By:** `"productLine"`

### Q38. Orders per customer
- **Tables:** `customers c`, `orders o` · **Joins:** `LEFT JOIN orders o ON o."customerNumber" = c."customerNumber"`
- **Columns:** `c."customerNumber"`, `c."customerName"`, `COUNT(o."orderNumber") AS "orderCount"`
- **Group By:** `c."customerNumber"`, `c."customerName"`

### Q39. Max MSRP per product line
- **Tables:** `products` · **Columns:** `"productLine"`, `MAX("MSRP") AS "maxMSRP"` · **Group By:** `"productLine"`

### Q40. Min buy price per vendor
- **Tables:** `products` · **Columns:** `"productVendor"`, `MIN("buyPrice") AS "minBuyPrice"` · **Group By:** `"productVendor"`

---

## Section D — Whole-table aggregates (Q41–Q50)

| ID | Question | Intent | Tables | Columns |
|----|----------|--------|--------|---------|
| 41 | Total number of customers | Whole-table COUNT | `customers` | `COUNT(*) AS "totalCustomers"` |
| 42 | Total number of products | Whole-table COUNT | `products` | `COUNT(*) AS "totalProducts"` |
| 43 | Total revenue from payments | Whole-table SUM | `payments` | `SUM("amount") AS "totalRevenue"` |
| 44 | Average product price | Whole-table AVG (MSRP) | `products` | `ROUND(AVG("MSRP"), 2) AS "avgMSRP"` |
| 45 | Max payment amount | Whole-table MAX | `payments` | `MAX("amount") AS "maxPayment"` |
| 46 | Min payment amount | Whole-table MIN | `payments` | `MIN("amount") AS "minPayment"` |
| 47 | Count total orders | Whole-table COUNT | `orders` | `COUNT(*) AS "totalOrders"` |
| 48 | Total quantity in stock | Whole-table SUM | `products` | `SUM("quantityInStock") AS "totalStock"` |
| 49 | Average MSRP | Whole-table AVG | `products` | `ROUND(AVG("MSRP"), 2) AS "avgMSRP"` |
| 50 | Number of employees | Whole-table COUNT | `employees` | `COUNT(*) AS "totalEmployees"` |

All ten questions in this section share the same shape:

```
Tables:    [single table]
Columns:   [single aggregate expression]
Filters:   none
Joins:     none
Group By:  none
```

---

## How decomposition becomes SQL (preview of Task 3)

Given any decomposition `D`, the Task 3 pipeline renders SQL with this template:

```
SELECT [DISTINCT] {D.columns}
FROM   {D.tables[0]}
       {D.joins …}
[WHERE {D.filters …}]
[GROUP BY {D.group_by …}]
[ORDER BY {D.order_by …}]
```

The fact that exactly the same template handles all 50 questions is the proof that this decomposition format captures the structure of the task correctly.

The machine-readable version of every decomposition above is in [`Task2_Decomposition.json`](./Task2_Decomposition.json).
