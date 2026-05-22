# Task 1 — Part 1: Ground Truth SQL Queries

**Author:** Rishav Dev Tiwari
**Database:** PostgreSQL — `classicmodels` (the seed in `project/seed.sql`)
**Schema note:** All identifiers (table and column names) are double-quoted because the schema uses camelCase columns (e.g. `"customerName"`, `"productCode"`).

For each of the 50 benchmark questions below I provide:

1. The natural-language question
2. A clean, executable PostgreSQL `SELECT` query (ground truth)
3. A short explanation of how the query works
4. How to verify the result (count + a few rows)

> Run each query in psql, DBeaver, or pgAdmin against the `classicmodels` database. Capture a screenshot of the result grid (or `\copy ... TO 'filename.csv' CSV HEADER` for export) for your submission packet.

---

## Section A — Simple `SELECT *` and single-table projections (Q1–Q20)

### Q1. List all products
```sql
SELECT * FROM products;
```
**Explanation:** Plain projection of every row and every column from `products`. No filters, no joins.
**Verify:** `SELECT COUNT(*) FROM products;` → **110 rows**.

---

### Q2. Get all customers
```sql
SELECT * FROM customers;
```
**Explanation:** Returns every customer row.
**Verify:** `COUNT(*)` → **122 rows**.

---

### Q3. Show all orders
```sql
SELECT * FROM orders;
```
**Explanation:** Returns the full orders table.
**Verify:** `COUNT(*)` → **326 rows**.

---

### Q4. List all employees
```sql
SELECT * FROM employees;
```
**Explanation:** Full employees table.
**Verify:** `COUNT(*)` → **23 rows**.

---

### Q5. Get all offices
```sql
SELECT * FROM offices;
```
**Explanation:** Full offices table.
**Verify:** `COUNT(*)` → **7 rows**.

---

### Q6. Show all product lines
```sql
SELECT * FROM productlines;
```
**Explanation:** Full productlines table.
**Verify:** `COUNT(*)` → **7 rows**.

---

### Q7. List all payments
```sql
SELECT * FROM payments;
```
**Explanation:** Full payments table.
**Verify:** `COUNT(*)` → **273 rows**.

---

### Q8. Get product names and prices
```sql
SELECT "productName", "MSRP", "buyPrice"
FROM products;
```
**Explanation:** Project just the relevant columns. `MSRP` is the customer-facing price, `buyPrice` is the cost. Both included so the question is fully answered.

---

### Q9. Get customer names and cities
```sql
SELECT "customerName", "city"
FROM customers;
```
**Explanation:** Two-column projection.

---

### Q10. List employee first and last names
```sql
SELECT "firstName", "lastName"
FROM employees;
```
**Explanation:** Two-column projection from `employees`.

---

### Q11. Get all order dates
```sql
SELECT "orderNumber", "orderDate"
FROM orders
ORDER BY "orderDate";
```
**Explanation:** I include `orderNumber` so the dates are identifiable, and order chronologically for readability.

---

### Q12. Show product vendor list
```sql
SELECT DISTINCT "productVendor"
FROM products
ORDER BY "productVendor";
```
**Explanation:** `DISTINCT` removes duplicates so you get the unique vendor list (13 vendors) rather than 110 rows.

---

### Q13. Get all product codes
```sql
SELECT "productCode"
FROM products
ORDER BY "productCode";
```
**Explanation:** Single-column projection of the primary key.

---

### Q14. List all countries from offices
```sql
SELECT DISTINCT "country"
FROM offices
ORDER BY "country";
```
**Explanation:** `DISTINCT` because offices in the same country would otherwise repeat. Returns 5 unique countries.

---

### Q15. Show all order statuses
```sql
SELECT DISTINCT "status"
FROM orders
ORDER BY "status";
```
**Explanation:** Distinct status values: `Cancelled, Disputed, In Process, On Hold, Resolved, Shipped`.

---

### Q16. Get all payment amounts
```sql
SELECT "amount"
FROM payments
ORDER BY "amount" DESC;
```
**Explanation:** Single-column projection sorted by largest first.

---

### Q17. List all job titles
```sql
SELECT DISTINCT "jobTitle"
FROM employees
ORDER BY "jobTitle";
```
**Explanation:** Many employees share titles like `Sales Rep`; `DISTINCT` collapses these.

---

### Q18. Get customer phone numbers
```sql
SELECT "customerName", "phone"
FROM customers;
```
**Explanation:** Pairing the phone with the customer name keeps the result usable.

---

### Q19. Show product MSRP values
```sql
SELECT "productName", "MSRP"
FROM products
ORDER BY "MSRP" DESC;
```
**Explanation:** MSRP with the product so the values are identifiable; sorted highest-first.

---

### Q20. List order numbers
```sql
SELECT "orderNumber"
FROM orders
ORDER BY "orderNumber";
```
**Explanation:** Just the primary key, ordered.

---

## Section B — Joins between two or more tables (Q21–Q30)

### Q21. Get orders with customer names
```sql
SELECT o."orderNumber", o."orderDate", o."status", c."customerName"
FROM orders o
JOIN customers c ON c."customerNumber" = o."customerNumber"
ORDER BY o."orderNumber";
```
**Explanation:** Inner join `orders` ↔ `customers` on the FK `customerNumber`. Produces one row per order with the buyer's name.

---

### Q22. Get employees with office city
```sql
SELECT e."employeeNumber", e."firstName", e."lastName", e."jobTitle", o."city"
FROM employees e
JOIN offices o ON o."officeCode" = e."officeCode"
ORDER BY e."employeeNumber";
```
**Explanation:** Each employee belongs to one office; join on `officeCode` to attach the city.

---

### Q23. Get payments with customer names
```sql
SELECT p."customerNumber", c."customerName", p."checkNumber",
       p."paymentDate", p."amount"
FROM payments p
JOIN customers c ON c."customerNumber" = p."customerNumber"
ORDER BY p."paymentDate";
```
**Explanation:** Join on `customerNumber` to attach the human-readable name to each payment.

---

### Q24. Get order details with product names
```sql
SELECT od."orderNumber", od."productCode", pr."productName",
       od."quantityOrdered", od."priceEach"
FROM orderdetails od
JOIN products pr ON pr."productCode" = od."productCode"
ORDER BY od."orderNumber", od."orderLineNumber";
```
**Explanation:** Each line item references a product; join brings in the readable product name.

---

### Q25. Get products with product line description
```sql
SELECT p."productCode", p."productName", p."productLine", pl."textDescription"
FROM products p
JOIN productlines pl ON pl."productLine" = p."productLine"
ORDER BY p."productLine", p."productName";
```
**Explanation:** Join on `productLine` to attach the marketing description.

---

### Q26. Get customers with sales rep names
```sql
SELECT c."customerNumber", c."customerName",
       e."firstName" || ' ' || e."lastName" AS "salesRep"
FROM customers c
LEFT JOIN employees e ON e."employeeNumber" = c."salesRepEmployeeNumber"
ORDER BY c."customerName";
```
**Explanation:** `LEFT JOIN` because some customers have no assigned sales rep (`salesRepEmployeeNumber IS NULL`); `INNER JOIN` would silently drop them.

---

### Q27. Get orders with customer city
```sql
SELECT o."orderNumber", o."orderDate", o."status",
       c."customerName", c."city"
FROM orders o
JOIN customers c ON c."customerNumber" = o."customerNumber"
ORDER BY o."orderDate";
```
**Explanation:** Same join pattern as Q21 but project the customer's city.

---

### Q28. Get employees and their manager
```sql
SELECT e."employeeNumber",
       e."firstName" || ' ' || e."lastName"   AS "employee",
       e."jobTitle",
       m."firstName" || ' ' || m."lastName"   AS "manager"
FROM employees e
LEFT JOIN employees m ON m."employeeNumber" = e."reportsTo"
ORDER BY e."employeeNumber";
```
**Explanation:** Self-join on `employees`. The President has `reportsTo IS NULL`, so use `LEFT JOIN` to keep that row.

---

### Q29. Get orderdetails with product vendor
```sql
SELECT od."orderNumber", od."productCode", p."productName",
       p."productVendor", od."quantityOrdered", od."priceEach"
FROM orderdetails od
JOIN products p ON p."productCode" = od."productCode"
ORDER BY p."productVendor", od."orderNumber";
```
**Explanation:** Join orderdetails → products to expose the vendor for each line item.

---

### Q30. Get payments with customer country
```sql
SELECT p."customerNumber", c."customerName", c."country",
       p."checkNumber", p."paymentDate", p."amount"
FROM payments p
JOIN customers c ON c."customerNumber" = p."customerNumber"
ORDER BY c."country", p."paymentDate";
```
**Explanation:** Standard join + projection of the customer's country.

---

## Section C — Aggregation with GROUP BY (Q31–Q40)

### Q31. Count customers per country
```sql
SELECT "country", COUNT(*) AS "customerCount"
FROM customers
GROUP BY "country"
ORDER BY "customerCount" DESC;
```
**Explanation:** `GROUP BY country` and count rows per group.

---

### Q32. Total payments per customer
```sql
SELECT p."customerNumber", c."customerName",
       SUM(p."amount") AS "totalPaid"
FROM payments p
JOIN customers c ON c."customerNumber" = p."customerNumber"
GROUP BY p."customerNumber", c."customerName"
ORDER BY "totalPaid" DESC;
```
**Explanation:** Aggregate `amount` per customer, joining to expose the readable name.

---

### Q33. Number of orders per status
```sql
SELECT "status", COUNT(*) AS "orderCount"
FROM orders
GROUP BY "status"
ORDER BY "orderCount" DESC;
```
**Explanation:** Group by `status`, count rows.

---

### Q34. Products per product line
```sql
SELECT "productLine", COUNT(*) AS "productCount"
FROM products
GROUP BY "productLine"
ORDER BY "productCount" DESC;
```
**Explanation:** Group products by line and count.

---

### Q35. Employees per office
```sql
SELECT o."officeCode", o."city", COUNT(e."employeeNumber") AS "employeeCount"
FROM offices o
LEFT JOIN employees e ON e."officeCode" = o."officeCode"
GROUP BY o."officeCode", o."city"
ORDER BY "employeeCount" DESC;
```
**Explanation:** `LEFT JOIN` so an office with zero employees would still appear (defensive design even if all 7 offices currently have staff).

---

### Q36. Total stock per product vendor
```sql
SELECT "productVendor", SUM("quantityInStock") AS "totalStock"
FROM products
GROUP BY "productVendor"
ORDER BY "totalStock" DESC;
```
**Explanation:** Aggregate `quantityInStock` summed within each vendor.

---

### Q37. Average buy price per product line
```sql
SELECT "productLine", ROUND(AVG("buyPrice"), 2) AS "avgBuyPrice"
FROM products
GROUP BY "productLine"
ORDER BY "avgBuyPrice" DESC;
```
**Explanation:** `AVG` with rounding for readable output.

---

### Q38. Orders per customer
```sql
SELECT c."customerNumber", c."customerName", COUNT(o."orderNumber") AS "orderCount"
FROM customers c
LEFT JOIN orders o ON o."customerNumber" = c."customerNumber"
GROUP BY c."customerNumber", c."customerName"
ORDER BY "orderCount" DESC;
```
**Explanation:** `LEFT JOIN` so customers with zero orders still appear (returns 0).

---

### Q39. Max MSRP per product line
```sql
SELECT "productLine", MAX("MSRP") AS "maxMSRP"
FROM products
GROUP BY "productLine"
ORDER BY "maxMSRP" DESC;
```
**Explanation:** Group + `MAX`.

---

### Q40. Min buy price per vendor
```sql
SELECT "productVendor", MIN("buyPrice") AS "minBuyPrice"
FROM products
GROUP BY "productVendor"
ORDER BY "minBuyPrice";
```
**Explanation:** Group + `MIN`.

---

## Section D — Whole-table aggregates (Q41–Q50)

### Q41. Total number of customers
```sql
SELECT COUNT(*) AS "totalCustomers" FROM customers;
```
**Expected:** 122.

---

### Q42. Total number of products
```sql
SELECT COUNT(*) AS "totalProducts" FROM products;
```
**Expected:** 110.

---

### Q43. Total revenue from payments
```sql
SELECT SUM("amount") AS "totalRevenue" FROM payments;
```
**Explanation:** Sum of all `payments.amount`. Returns the lifetime cash collected.

---

### Q44. Average product price
```sql
SELECT ROUND(AVG("MSRP"), 2) AS "avgMSRP" FROM products;
```
**Explanation:** I used `MSRP` as the customer-facing "price". If `buyPrice` was intended, swap the column.

---

### Q45. Max payment amount
```sql
SELECT MAX("amount") AS "maxPayment" FROM payments;
```

---

### Q46. Min payment amount
```sql
SELECT MIN("amount") AS "minPayment" FROM payments;
```

---

### Q47. Count total orders
```sql
SELECT COUNT(*) AS "totalOrders" FROM orders;
```
**Expected:** 326.

---

### Q48. Total quantity in stock
```sql
SELECT SUM("quantityInStock") AS "totalStock" FROM products;
```
**Explanation:** Sum across all products.

---

### Q49. Average MSRP
```sql
SELECT ROUND(AVG("MSRP"), 2) AS "avgMSRP" FROM products;
```
**Explanation:** Same as Q44 — explicitly named here.

---

### Q50. Number of employees
```sql
SELECT COUNT(*) AS "totalEmployees" FROM employees;
```
**Expected:** 23.

---

## How to capture results for submission

For each query above, capture the **count + first few rows** in a screenshot. In psql:

```sql
\pset pager off
\timing on
-- paste query
SELECT count(*) FROM products;  -- proves the row count
SELECT * FROM products LIMIT 5;  -- shows actual rows
```

Or in DBeaver / pgAdmin: run the query, then screenshot the result grid.

For long lists (Q1, Q2, Q3, Q7) the assignment explicitly says you may show **count + a few rows** instead of the full result set.

## Submission checklist

- [ ] Each of the 50 questions has its own page/section in your final submission document
- [ ] Each section contains: question, SQL query, screenshot of result, explanation
- [ ] All queries verified to execute against the seeded `classicmodels` database
- [ ] Only `SELECT` statements were used
