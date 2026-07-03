"""
schemas.py
==========

Pydantic request/response schemas for the Office Attendance Management
System FastAPI backend.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ===========================================================================
# Auth
# ===========================================================================

class Token(BaseModel):
    """Response body for POST /token."""

    access_token: str
    token_type: str = "bearer"
    role: str
    employee_id: int
    name: str


class TokenData(BaseModel):
    """Claims embedded inside a decoded JWT."""

    employee_id: Optional[int] = None
    role: Optional[str] = None


# ===========================================================================
# Employee
# ===========================================================================

class EmployeeBase(BaseModel):
    name: str
    department: Optional[str] = None


class EmployeeCreate(EmployeeBase):
    """Body for POST /employees/ (admin-only registration)."""

    password: str
    face_encoding: List[float]
    role: str = "employee"  # allow an admin to register another admin


class EmployeeOut(EmployeeBase):
    id: int
    role: str
    #: Stored as a JSON string in the DB; dashboard.py already handles
    #: str-or-list on the way in, so it's returned as-is here.
    face_encoding: Optional[str] = None

    class Config:
        from_attributes = True


# ===========================================================================
# Attendance
# ===========================================================================

class AttendanceLogCreate(BaseModel):
    """Body for POST /attendance/log (called by the kiosk, unauthenticated)."""

    employee_id: int
    status: str = Field(pattern="^(Present|Late)$")


class AttendanceLogOut(BaseModel):
    id: int
    employee_id: int
    timestamp: datetime
    status: str

    class Config:
        from_attributes = True


class AttendanceStats(BaseModel):
    """Response body for GET /attendance/stats/me."""

    employee_id: int
    total_days_logged: int
    present_count: int
    late_count: int
    attendance_rate: float  # present_count / total_days_logged, in [0, 1]


# ===========================================================================
# Leave
# ===========================================================================

class LeaveRequestCreate(BaseModel):
    """Body for POST /leaves/ (any authenticated employee, applies for themselves)."""

    start_date: datetime
    end_date: datetime
    reason: Optional[str] = None


class LeaveRequestOut(BaseModel):
    id: int
    employee_id: int
    employee_name: Optional[str] = None
    start_date: datetime
    end_date: datetime
    status: str
    reason: Optional[str] = None

    class Config:
        from_attributes = True


class LeaveRequestUpdate(BaseModel):
    """Body for PUT /leaves/{id} (admin-only approve/reject)."""

    status: str = Field(pattern="^(Approved|Rejected)$")


# ===========================================================================
# Intruder
# ===========================================================================

class IntruderCreate(BaseModel):
    """Body for POST /intruders/log (called by IntruderManager, unauthenticated)."""

    image_path: str


class IntruderOut(BaseModel):
    id: int
    image_path: str
    timestamp: datetime

    class Config:
        from_attributes = True
