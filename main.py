"""
main.py
=======

FastAPI backend for the enterprise Office Attendance Management System.

Phase 7: adds JWT authentication and role-based access control
(admin vs employee) on top of the Employee / AttendanceLog / LeaveRequest
/ Intruder surface that ``vision_engine.py``, ``dashboard.py``, and
``system_utilities.py`` were already written against.

New dependencies for this phase:
    pip install "passlib[bcrypt]" "python-jose[cryptography]" python-multipart

("python-multipart" is required by FastAPI's OAuth2PasswordRequestForm,
which parses the application/x-www-form-urlencoded body a login form
submits -- without it, POST /token will fail at import time.)

Run with:
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload
or via run_system.py, which launches this alongside dashboard.py.

Security notes (read before deploying beyond your own laptop)
---------------------------------------------------------------
* Change ``auth.SECRET_KEY`` -- see the warning in ``auth.py``.
* CORS is wide open (``allow_origins=["*"]``) below so both the Streamlit
  dashboard and the Expo mobile app can reach this API during development.
  Restrict it to known origins before deploying.
* ``POST /attendance/log`` and ``POST /intruders/log`` are intentionally
  left WITHOUT an auth dependency: they're called by the unattended kiosk
  (dashboard.py's camera loop), not by a logged-in employee's own browser
  session, so there's no per-employee JWT to check there. As shipped,
  that means anyone who can reach the API can post fake attendance/intruder
  events. For real deployment, protect these with a separate kiosk API key
  (e.g. a static header checked by a small dependency) rather than leaving
  them fully open.
"""

from __future__ import annotations

from datetime import timedelta
from typing import List

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

import auth
import crud
import models
import schemas
from database import engine, get_db

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Office Attendance Management System API",
    description="JWT-secured, role-based backend for face-recognition attendance tracking.",
    version="7.0.0",
)

# Both the Streamlit dashboard (different port, localhost) and the Expo
# mobile app (a phone on the same Wi-Fi, different IP) call this API from
# different origins, so CORS is left open for local development. Tighten
# this (allow_origins=["http://localhost:8501", ...]) before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# tokenUrl is relative to the API root -- this is what makes FastAPI's
# auto-generated /docs "Authorize" button POST to /token automatically.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Office Attendance API</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background:#0f172a;
                   color:#f8fafc; display:flex; flex-direction:column; align-items:center;
                   justify-content:center; height:100vh; margin:0; }
            .card { background:#1e293b; padding:2.5rem; border-radius:1rem; text-align:center;
                     max-width:520px; border:1px solid #334155; }
            h1 { color:#38bdf8; margin-top:0; }
            p { color:#94a3b8; line-height:1.6; }
            a.btn { display:inline-block; background:linear-gradient(135deg,#38bdf8,#0ea5e9); color:#fff;
                     padding:.75rem 1.5rem; text-decoration:none; border-radius:.5rem; font-weight:600;
                     margin:.4rem; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Office Attendance API 🕒</h1>
            <p>JWT-secured backend for the kiosk, admin dashboard, and employee mobile app.</p>
            <a class="btn" href="/docs">API Docs</a>
        </div>
    </body>
    </html>
    """


# ===========================================================================
# Auth dependencies
# ===========================================================================

def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> models.Employee:
    """Resolve the JWT bearer token on a request into the Employee it belongs to.

    Use as ``current_user: models.Employee = Depends(get_current_user)`` on
    any endpoint that requires *some* logged-in employee (any role).

    Raises:
        HTTPException 401: if the token is missing, malformed, expired, or
            refers to an employee that no longer exists.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = auth.decode_access_token(token)
    if payload is None:
        raise credentials_exception

    employee_id = payload.get("employee_id")
    if employee_id is None:
        raise credentials_exception

    employee = crud.get_employee(db, employee_id)
    if employee is None:
        raise credentials_exception
    return employee


def require_admin(current_user: models.Employee = Depends(get_current_user)) -> models.Employee:
    """Dependency that additionally enforces ``role == "admin"``.

    Use in place of ``get_current_user`` on admin-only endpoints. Because
    it depends on ``get_current_user``, an invalid/missing token still
    correctly produces a 401 before the 403 role check ever runs.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required"
        )
    return current_user


# ===========================================================================
# Auth endpoints
# ===========================================================================

@app.post("/token", response_model=schemas.Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
) -> schemas.Token:
    """Standard OAuth2 password-flow login.

    ``form_data.username`` is matched against ``Employee.name``. If you
    want login identifiers separate from display names, add a dedicated
    ``username``/``email`` column to the Employee model and match on that
    instead.

    Both dashboard.py and the mobile App.js POST here with
    application/x-www-form-urlencoded bodies (NOT JSON) -- that's a
    requirement of OAuth2PasswordRequestForm, not a choice either client
    makes independently.
    """
    employee = crud.get_employee_by_name(db, form_data.username)
    if not employee or not auth.verify_password(form_data.password, employee.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = auth.create_access_token(
        data={"employee_id": employee.id, "role": employee.role},
        expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return schemas.Token(
        access_token=access_token,
        token_type="bearer",
        role=employee.role,
        employee_id=employee.id,
        name=employee.name,
    )


# ===========================================================================
# Employee endpoints
# ===========================================================================

@app.post("/employees/", response_model=schemas.EmployeeOut, status_code=status.HTTP_201_CREATED)
def create_employee(
    employee: schemas.EmployeeCreate,
    db: Session = Depends(get_db),
    _admin: models.Employee = Depends(require_admin),
):
    """Register a new employee (face embedding + login credentials). Admin-only."""
    if crud.get_employee_by_name(db, employee.name):
        raise HTTPException(status_code=400, detail="An employee with this name already exists")
    return crud.create_employee(db, employee)


@app.get("/employees/", response_model=List[schemas.EmployeeOut])
def read_employees(
    db: Session = Depends(get_db), _admin: models.Employee = Depends(require_admin)
):
    """List every employee, including their face embedding. Admin-only --
    this is what dashboard.py's kiosk page uses to build known_embeddings."""
    return crud.get_employees(db)


@app.get("/employees/me", response_model=schemas.EmployeeOut)
def read_current_employee(current_user: models.Employee = Depends(get_current_user)):
    """The logged-in employee's own profile."""
    return current_user


# ===========================================================================
# Attendance endpoints
# ===========================================================================

@app.post(
    "/attendance/log", response_model=schemas.AttendanceLogOut, status_code=status.HTTP_201_CREATED
)
def log_attendance(log: schemas.AttendanceLogCreate, db: Session = Depends(get_db)):
    """Log one check-in event. Called by the kiosk (dashboard.py), which is
    an unattended camera loop, not a specific employee's browser session --
    hence no auth dependency here. See the module-level security note."""
    if crud.get_employee(db, log.employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    return crud.create_attendance_log(db, log)


@app.get("/attendance/stats/me", response_model=schemas.AttendanceStats)
def read_my_attendance_stats(
    db: Session = Depends(get_db), current_user: models.Employee = Depends(get_current_user)
):
    """Personal attendance stats for the logged-in employee. Used by both
    the Streamlit employee view and the mobile app's Home screen."""
    return crud.get_attendance_stats(db, current_user.id)


# ===========================================================================
# Leave management endpoints
# ===========================================================================

@app.post("/leaves/", response_model=schemas.LeaveRequestOut, status_code=status.HTTP_201_CREATED)
def apply_for_leave(
    leave: schemas.LeaveRequestCreate,
    db: Session = Depends(get_db),
    current_user: models.Employee = Depends(get_current_user),
):
    """Any authenticated employee can apply for their own leave (web or mobile)."""
    db_leave = crud.create_leave_request(db, current_user.id, leave)
    return schemas.LeaveRequestOut(
        id=db_leave.id,
        employee_id=db_leave.employee_id,
        employee_name=current_user.name,
        start_date=db_leave.start_date,
        end_date=db_leave.end_date,
        status=db_leave.status,
        reason=db_leave.reason,
    )


@app.get("/leaves/me", response_model=List[schemas.LeaveRequestOut])
def read_my_leaves(
    db: Session = Depends(get_db), current_user: models.Employee = Depends(get_current_user)
):
    """The logged-in employee's own leave request history."""
    leaves = crud.get_leaves_for_employee(db, current_user.id)
    return [
        schemas.LeaveRequestOut(
            id=leave.id,
            employee_id=leave.employee_id,
            employee_name=current_user.name,
            start_date=leave.start_date,
            end_date=leave.end_date,
            status=leave.status,
            reason=leave.reason,
        )
        for leave in leaves
    ]


@app.get("/leaves/pending", response_model=List[schemas.LeaveRequestOut])
def read_pending_leaves(
    db: Session = Depends(get_db), _admin: models.Employee = Depends(require_admin)
):
    """Admin-only: every leave request still awaiting a decision."""
    leaves = crud.get_pending_leaves(db)
    return [_leave_to_out(leave) for leave in leaves]


@app.get("/leaves/", response_model=List[schemas.LeaveRequestOut])
def read_all_leaves(
    db: Session = Depends(get_db), _admin: models.Employee = Depends(require_admin)
):
    """Admin-only: every leave request ever made, any status -- powers the
    Leave Dashboard's calendar view (needs Approved leaves too, not just Pending)."""
    leaves = crud.get_all_leaves(db)
    return [_leave_to_out(leave) for leave in leaves]


@app.put("/leaves/{leave_id}", response_model=schemas.LeaveRequestOut)
def update_leave(
    leave_id: int,
    update: schemas.LeaveRequestUpdate,
    db: Session = Depends(get_db),
    _admin: models.Employee = Depends(require_admin),
):
    """Admin-only: approve or reject a pending leave request."""
    db_leave = crud.update_leave_status(db, leave_id, update.status)
    if db_leave is None:
        raise HTTPException(status_code=404, detail="Leave request not found")
    return _leave_to_out(db_leave)


def _leave_to_out(leave: models.LeaveRequest) -> schemas.LeaveRequestOut:
    """Shared helper: LeaveRequest ORM row -> LeaveRequestOut, resolving the
    employee's name via the relationship so admin views don't need a
    separate employee lookup per row."""
    return schemas.LeaveRequestOut(
        id=leave.id,
        employee_id=leave.employee_id,
        employee_name=leave.employee.name if leave.employee else None,
        start_date=leave.start_date,
        end_date=leave.end_date,
        status=leave.status,
        reason=leave.reason,
    )


# ===========================================================================
# Intruder endpoints
# ===========================================================================

@app.post(
    "/intruders/log", response_model=schemas.IntruderOut, status_code=status.HTTP_201_CREATED
)
def log_intruder(intruder: schemas.IntruderCreate, db: Session = Depends(get_db)):
    """Log a suspected-intruder snapshot. Called by
    system_utilities.IntruderManager -- no auth, same reasoning as
    /attendance/log (the kiosk isn't a logged-in employee session)."""
    return crud.create_intruder_log(db, intruder)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
