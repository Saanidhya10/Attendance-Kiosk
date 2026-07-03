"""
models.py
=========

SQLAlchemy models for the Office Attendance Management System, covering
everything the rest of the system (``vision_engine.py``, ``dashboard.py``,
``system_utilities.py``) has been built against:

    Employee        -- enrolled staff, now with login credentials + role
    AttendanceLog   -- one row per kiosk check-in
    LeaveRequest    -- employee-submitted leave, admin-approved
    Intruder        -- suspected-intruder snapshots from IntruderManager
"""

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database import Base


class Employee(Base):
    """An enrolled employee: identity, face embedding, and login credentials."""

    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True, index=True)
    department = Column(String, nullable=True)

    #: JSON-encoded list[float] facial embedding (see vision_engine.register_face).
    face_encoding = Column(Text, nullable=True)

    hashed_password = Column(String, nullable=False)

    #: "employee" (default) or "admin". Checked by main.py's require_admin dependency.
    role = Column(String, nullable=False, default="employee")

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    attendance_logs = relationship(
        "AttendanceLog", back_populates="employee", cascade="all, delete-orphan"
    )
    leave_requests = relationship(
        "LeaveRequest", back_populates="employee", cascade="all, delete-orphan"
    )


class AttendanceLog(Base):
    """One kiosk check-in event for one employee."""

    __tablename__ = "attendance_logs"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String, nullable=False)  # "Present" | "Late"

    employee = relationship("Employee", back_populates="attendance_logs")


class LeaveRequest(Base):
    """An employee's leave request and its current approval status."""

    __tablename__ = "leave_requests"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False, index=True)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    status = Column(String, nullable=False, default="Pending")  # Pending|Approved|Rejected
    reason = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    employee = relationship("Employee", back_populates="leave_requests")


class Intruder(Base):
    """A snapshot logged by system_utilities.IntruderManager when an unknown
    face is seen for too many consecutive frames."""

    __tablename__ = "intruders"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
