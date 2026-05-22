"""
Static schema description of the ClassicModels database.

Used by:
- decomposer for rule-based table/column inference on unseen questions
- validator for column-existence checks during retry
- nl_answer for human-readable summaries
"""
from typing import Dict, List

# All table names use plural lowercase. Column names are camelCase
# and MUST be referenced with double quotes in PostgreSQL.
SCHEMA: Dict[str, List[str]] = {
    "productlines": [
        "productLine", "textDescription", "htmlDescription", "image",
    ],
    "products": [
        "productCode", "productName", "productLine", "productScale",
        "productVendor", "productDescription", "quantityInStock",
        "buyPrice", "MSRP",
    ],
    "offices": [
        "officeCode", "city", "phone", "addressLine1", "addressLine2",
        "state", "country", "postalCode", "territory",
    ],
    "employees": [
        "employeeNumber", "lastName", "firstName", "extension", "email",
        "officeCode", "reportsTo", "jobTitle",
    ],
    "customers": [
        "customerNumber", "customerName", "contactLastName",
        "contactFirstName", "phone", "addressLine1", "addressLine2",
        "city", "state", "postalCode", "country",
        "salesRepEmployeeNumber", "creditLimit",
    ],
    "payments": [
        "customerNumber", "checkNumber", "paymentDate", "amount",
    ],
    "orders": [
        "orderNumber", "orderDate", "requiredDate", "shippedDate",
        "status", "comments", "customerNumber",
    ],
    "orderdetails": [
        "orderNumber", "productCode", "quantityOrdered", "priceEach",
        "orderLineNumber",
    ],
}

# Foreign-key relationships, used by the rule-based fallback
# decomposer to wire up joins automatically.
FOREIGN_KEYS = [
    # (left_table, left_col, right_table, right_col)
    ("products",     "productLine",            "productlines", "productLine"),
    ("employees",    "officeCode",             "offices",      "officeCode"),
    ("employees",    "reportsTo",              "employees",    "employeeNumber"),
    ("customers",    "salesRepEmployeeNumber", "employees",    "employeeNumber"),
    ("orders",       "customerNumber",         "customers",    "customerNumber"),
    ("orderdetails", "orderNumber",            "orders",       "orderNumber"),
    ("orderdetails", "productCode",            "products",     "productCode"),
    ("payments",     "customerNumber",         "customers",    "customerNumber"),
]


def all_tables() -> List[str]:
    return list(SCHEMA.keys())


def all_columns_flat() -> List[str]:
    return sorted({c for cols in SCHEMA.values() for c in cols})


def column_exists(table: str, column: str) -> bool:
    return table in SCHEMA and column in SCHEMA[table]


def find_column_table(column: str) -> List[str]:
    """Return all tables that have a column with this exact (case-sensitive) name."""
    return [t for t, cols in SCHEMA.items() if column in cols]
