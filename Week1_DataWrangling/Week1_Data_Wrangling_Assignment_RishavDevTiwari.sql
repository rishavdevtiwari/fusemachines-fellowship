-- 1. Show all the customers whose creditLimit is greater than 20000
SELECT * FROM customers WHERE creditLimit > 20000;

-- 2. Show the employees who report to VP Sales
SELECT * FROM employees 
WHERE reportsTo = (SELECT employeeNumber FROM employees WHERE jobTitle = 'VP Sales');

-- 3. Find customers who set their state, live in USA, and credit limit is between 100k and 200k
SELECT * FROM customers 
WHERE state IS NOT NULL 
  AND country = 'USA' 
  AND creditLimit BETWEEN 100000 AND 200000;

-- 4. Find all employees who report to Sales Managers of all types
SELECT * FROM employees 
WHERE reportsTo IN (SELECT employeeNumber FROM employees WHERE jobTitle LIKE '%Sales Manager%');

-- 5. Find the average credit limit of customers of each country
SELECT country, AVG(creditLimit) AS average_credit_limit 
FROM customers 
GROUP BY country;

-- 6. Find total no. of orders for each date and customer (Total > 10)
SELECT orderDate, customerNumber, COUNT(orderNumber) AS total_orders 
FROM orders 
GROUP BY orderDate, customerNumber 
HAVING total_orders > 10;

-- 7. Find supervisor name, title, and total supervisees using a subquery (NO JOIN)
SELECT firstName, lastName, jobTitle, 
       (SELECT COUNT(*) FROM employees e2 WHERE e2.reportsTo = e1.employeeNumber) AS supervisee_count
FROM employees e1
WHERE employeeNumber IN (SELECT DISTINCT reportsTo FROM employees WHERE reportsTo IS NOT NULL);

-- 8. Find supervisor name, title, and total supervisees using a JOIN
SELECT e1.firstName, e1.lastName, e1.jobTitle, COUNT(e2.employeeNumber) AS supervisee_count
FROM employees e1
JOIN employees e2 ON e1.employeeNumber = e2.reportsTo
GROUP BY e1.employeeNumber;

-- 9. Find all customers with a credit limit greater than average credit limit using WITH Clause
WITH AvgCredit AS (
    SELECT AVG(creditLimit) AS avg_limit FROM customers
)
SELECT c.* 
FROM customers c, AvgCredit a
WHERE c.creditLimit > a.avg_limit;

-- 10. Find the rank of customer by credit limit, then find the 3rd highest
WITH RankedCustomers AS (
    SELECT customerName, creditLimit, 
           DENSE_RANK() OVER(ORDER BY creditLimit DESC) AS ranking 
    FROM customers
)
SELECT * FROM RankedCustomers WHERE ranking = 3;

-- 11. Generate a report showing total no. of employees working in each office
SELECT o.city, COUNT(e.employeeNumber) AS total_employees 
FROM offices o 
JOIN employees e ON o.officeCode = e.officeCode 
GROUP BY o.officeCode;

-- 12. Generate a report showing total no. of customers associated with each office
SELECT o.city, COUNT(c.customerNumber) AS total_customers 
FROM offices o 
JOIN employees e ON o.officeCode = e.officeCode 
JOIN customers c ON e.employeeNumber = c.salesRepEmployeeNumber 
GROUP BY o.officeCode;

-- 13. Generate a report showing total payment received by each office
SELECT o.city, o.state, o.country, SUM(p.amount) AS total_payments 
FROM offices o 
JOIN employees e ON o.officeCode = e.officeCode 
JOIN customers c ON e.employeeNumber = c.salesRepEmployeeNumber 
JOIN payments p ON c.customerNumber = p.customerNumber 
GROUP BY o.officeCode;

-- 14. Generate a report showing total sales (in amount) by each office
SELECT o.city, SUM(od.quantityOrdered * od.priceEach) AS total_sales 
FROM offices o 
JOIN employees e ON o.officeCode = e.officeCode 
JOIN customers c ON e.employeeNumber = c.salesRepEmployeeNumber 
JOIN orders ord ON c.customerNumber = ord.customerNumber 
JOIN orderdetails od ON ord.orderNumber = od.orderNumber 
GROUP BY o.officeCode;

-- 15. Generate a report showing total payment pending for each office (Using 'On Hold' status as pending)
SELECT o.city, SUM(od.quantityOrdered * od.priceEach) AS total_pending 
FROM offices o 
JOIN employees e ON o.officeCode = e.officeCode 
JOIN customers c ON e.employeeNumber = c.salesRepEmployeeNumber 
JOIN orders ord ON c.customerNumber = ord.customerNumber 
JOIN orderdetails od ON ord.orderNumber = od.orderNumber 
WHERE ord.status = 'On Hold'
GROUP BY o.officeCode;

-- 16. Find proportion of creditLimit of each person in each country
SELECT customerName, country, creditLimit, 
       creditLimit / SUM(creditLimit) OVER(PARTITION BY country) AS proportion_in_country 
FROM customers;

-- 17. Create a view showing customer name, complete address, and total orders
CREATE VIEW CustomerOrderSummary AS 
SELECT c.customerName, 
       CONCAT(c.addressLine1, ' ', IFNULL(c.addressLine2, ''), ', ', c.city, ', ', IFNULL(c.state, ''), ' ', IFNULL(c.postalCode, ''), ' ', c.country) AS full_address, 
       COUNT(o.orderNumber) AS total_orders 
FROM customers c 
LEFT JOIN orders o ON c.customerNumber = o.customerNumber 
GROUP BY c.customerNumber;

-- 18. Update the country of a customer
UPDATE customers SET country = 'United States' WHERE customerNumber = 112;

-- 19. Delete all payments below 20,000
DELETE FROM payments WHERE amount < 20000;

-- 20. Add new payments manually for an existing customer
INSERT INTO payments (customerNumber, checkNumber, paymentDate, amount) 
VALUES (112, 'CHK999999', '2023-10-25', 25000.00);