# app/loan_products.py

"""
This file acts as a centralized catalog for all loan products offered.
It defines the core attributes and eligibility rules for each product.
This allows for easy updates without changing the main service logic.
"""

LOAN_PRODUCTS_CATALOG = [
    {
        "product_id": "SALARY_LOAN_PUBLIC",
        "product_name": "ATM Salary Loan (Public Sector)",
        "interest_rate_monthly": 1.58,
        "max_loanable_amount": 100000,
        "max_term_months": 12,
        "eligibility_rules": {
            "is_new_client_eligible": True,
            "employment_sector": ["Public"],
            "job": [] # No specific job required
        }
    },
    {
        "product_id": "SALARY_LOAN_PRIVATE",
        "product_name": "ATM Salary Loan (Private Sector)",
        "interest_rate_monthly": 2.5,
        "max_loanable_amount": 50000,
        "max_term_months": 6,
        "eligibility_rules": {
            "is_new_client_eligible": True,
            "employment_sector": ["Private"],
            "job": []
        }
    },
    {
        "product_id": "TEACHER_LOAN",
        "product_name": "Teacher's Loan",
        "interest_rate_monthly": 1.58,
        "max_loanable_amount": 100000,
        "max_term_months": 12,
        "eligibility_rules": {
            "is_new_client_eligible": True,
            "employment_sector": ["Public"],
            "job": ["Teacher"] # Specific to teachers
        }
    },
    {
        "product_id": "SECURITY_GUARD_LOAN",
        "product_name": "Security Guard Loan",
        "interest_rate_monthly": 2.5,
        "max_loanable_amount": 50000, # Assuming same as private
        "max_term_months": 6,
        "eligibility_rules": {
            "is_new_client_eligible": True,
            "employment_sector": ["Private"],
            "job": ["Security Guard"]
        }
    },
    # --- Loans for EXISTING clients only ---
    {
        "product_id": "EMERGENCY_LOAN",
        "product_name": "Emergency Loan",
        "interest_rate_monthly": 2.5,
        "max_loanable_amount": 30000, # Example amount
        "max_term_months": 6,
        "eligibility_rules": {
            "is_new_client_eligible": False, # Must be an existing client
            "employment_sector": ["Public", "Private"],
            "job": []
        }
    },
    # Add other products like Graduation Loan, Bonus Loan here following the same structure.
]