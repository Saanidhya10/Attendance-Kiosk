"""
crud.py
=======

Database access functions for the Office Attendance Management System.
Kept free of any HTTP/FastAPI concerns -- ``main.py`` is the only module
that translates these into endpoints.
"""

from __future__ import annotations

import json
from typing import List, Optional

from sqlalchemy.orm import Session

import models
import schemas
from auth import get_password_hash


# ===========================================================================
# Employee
# ===========================================================================

def get_employee(db: Session, employee_id: int) -> Optional[models.Employee]:
    return db.query(models.Employee).filter(models.Employee.id == employee_id).first()


def get_employee_by_name(db: Session, name: str) -> Optional[models.Employee]:
    return db.query(models.Employee).filter(models.Employee.name == name).first()


def get_employees(db: Session, skip: int = 0, limit: int = 500) -> List[models.Employee]:
    return db.query(models.Employee).offset(skip).limit(limit).all()


def create_employee(db: Session, employee: schemas.EmployeeCreate) -> models.Employee:
    """Create an employee, hashing their password and JSON-encoding their embedding."""
    db_employee = models.Employee(
        name=employee.name,
        department=employee.department,
        face_encoding=json.dumps(employee.face_encoding),
        hashed_password=get_password_hash(employee.password),
        role=employee.role,
    )
    db.add(db_employee)
    db.commit()
    db.refresh(db_employee)
    return db_employee


# ===========================================================================
# Attendance
# ===========================================================================

def create_attendance_log(db: Session, log: schemas.AttendanceLogCreate) -> models.AttendanceLog:
    db_log = models.AttendanceLog(employee_id=log.employee_id, status=log.status)
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log


def get_attendance_stats(db: Session, employee_id: int) -> schemas.AttendanceStats:
    """Aggregate one employee's attendance history into summary stats."""
    logs = (
        db.query(models.AttendanceLog)
        .filter(models.AttendanceLog.employee_id == employee_id)
        .all()
    )
    total = len(logs)
    present = sum(1 for log in logs if log.status == "Present")
    late = sum(1 for log in logs if log.status == "Late")
    rate = (present / total) if total else 0.0
    return schemas.AttendanceStats(
        employee_id=employee_id,
        total_days_logged=total,
        present_count=present,
        late_count=late,
        attendance_rate=rate,
    )


# ===========================================================================
# Leave
# ===========================================================================

def create_leave_request(
    db: Session, employee_id: int, leave: schemas.LeaveRequestCreate
) -> models.LeaveRequest:
    db_leave = models.LeaveRequest(
        employee_id=employee_id,
        start_date=leave.start_date,
        end_date=leave.end_date,
        reason=leave.reason,
        status="Pending",
    )
    db.add(db_leave)
    db.commit()
    db.refresh(db_leave)
    return db_leave


def get_pending_leaves(db: Session) -> List[models.LeaveRequest]:
    return db.query(models.LeaveRequest).filter(models.LeaveRequest.status == "Pending").all()


def get_all_leaves(db: Session) -> List[models.LeaveRequest]:
    return db.query(models.LeaveRequest).order_by(models.LeaveRequest.start_date.desc()).all()


def get_leaves_for_employee(db: Session, employee_id: int) -> List[models.LeaveRequest]:
    return (
        db.query(models.LeaveRequest)
        .filter(models.LeaveRequest.employee_id == employee_id)
        .order_by(models.LeaveRequest.start_date.desc())
        .all()
    )


def get_leave(db: Session, leave_id: int) -> Optional[models.LeaveRequest]:
    return db.query(models.LeaveRequest).filter(models.LeaveRequest.id == leave_id).first()


def update_leave_status(db: Session, leave_id: int, status: str) -> Optional[models.LeaveRequest]:
    db_leave = get_leave(db, leave_id)
    if not db_leave:
        return None
    db_leave.status = status
    db.commit()
    db.refresh(db_leave)
    return db_leave


# ===========================================================================
# Intruder
# ===========================================================================

def create_intruder_log(db: Session, intruder: schemas.IntruderCreate) -> models.Intruder:
    db_intruder = models.Intruder(image_path=intruder.image_path)
    db.add(db_intruder)
    db.commit()
    db.refresh(db_intruder)
    return db_intruder
