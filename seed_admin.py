"""
seed_admin.py
=============
One-time setup script that creates the very first admin account directly
in the database, bypassing the API entirely.

Why this exists: ``POST /employees/`` (the only way to create an account)
requires a valid *admin* JWT -- which means there's no way to create the
first admin through the API itself. That's normal for RBAC systems (the
same problem Django solves with ``createsuperuser``); this script is this
project's equivalent.

Run this ONCE, before starting the backend for the first time:
    python seed_admin.py

It prompts for a name and password interactively (no face photo -- an
admin created this way can log into the dashboard/mobile app immediately,
but won't be recognized by the kiosk camera until you also enroll them
with a photo via the Employee Management page, same as any other hire).

After this, log in via the Streamlit dashboard or mobile app with those
credentials, then use Employee Management to register everyone else
normally -- that flow works fine once at least one admin token exists.
"""

import getpass
import sys

import crud
import models
import schemas
from database import Base, SessionLocal, engine


def main() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        existing_admins = (
            db.query(models.Employee).filter(models.Employee.role == "admin").count()
        )
        if existing_admins > 0:
            print(
                f"There's already at least one admin account ({existing_admins} found). "
                f"Log in with an existing admin and use Employee Management instead."
            )
            sys.exit(0)

        print("=== Create the first admin account ===")
        name = input("Full name: ").strip()
        department = input("Department [Admin]: ").strip() or "Admin"
        password = getpass.getpass("Password: ")
        confirm = getpass.getpass("Confirm password: ")

        if not name or not password:
            print("Name and password are both required.")
            sys.exit(1)
        if password != confirm:
            print("Passwords didn't match -- nothing was created.")
            sys.exit(1)
        if crud.get_employee_by_name(db, name):
            print(f"An employee named {name!r} already exists.")
            sys.exit(1)

        employee = schemas.EmployeeCreate(
            name=name,
            department=department,
            password=password,
            role="admin",
            face_encoding=[],  # no kiosk photo yet -- see module docstring
        )
        db_employee = crud.create_employee(db, employee)
        print(f"\n✅ Admin account created: {db_employee.name} (id={db_employee.id})")
        print("You can now log in via the Streamlit dashboard or the mobile app.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
