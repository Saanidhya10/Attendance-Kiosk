"""
dashboard.py
============

Phase 3/6/7 -- Streamlit control panel for the enterprise Office
Attendance Management System. Wraps the CV engine (``vision_engine.py``)
and the FastAPI backend (``main.py``) into a single, role-based portal.

The app now starts on a **Login screen** (POST /token) and shows a
different sidebar depending on the logged-in employee's role:

    Admin:
        1. Live Camera Feed    -- Group Recognition kiosk
        2. Employee Management -- register new employees
        3. Analytics & Reports -- charts + raw log table read from SQLite
        4. Leave Dashboard     -- approve/reject leave, calendar view

    Employee:
        1. My Dashboard   -- personal attendance rate + late days
        2. Apply for Leave -- submit a leave request, see its history

Auth model
----------
On login, the JWT + role + employee identity are stored in
``st.session_state`` (NOT on disk -- closing the browser tab logs you
out, matching typical web-app session behavior). Every authenticated
backend call goes through ``authenticated_request()`` below, which
attaches the bearer token and, on a 401 (expired/invalid token), logs the
user out and sends them back to the login screen automatically.

Two endpoints are deliberately called WITHOUT a token: POST
/attendance/log and (implicitly, inside vision_engine's IntruderManager)
POST /intruders/log. Both are fired by the unattended kiosk loop, not by
a specific logged-in employee's session -- see main.py's module
docstring for the security tradeoff that implies.

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Union

import cv2
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from vision_engine import (
    FrameSkipper,
    InvalidImageError,
    NoFaceDetectedError,
    detect_liveness_multi,
    recognize_multiple_faces,
    register_face,
    reset_multi_liveness_state,
)

# ---------------------------------------------------------------------------
# CONFIG -- adjust to match your real backend/db if it differs
# ---------------------------------------------------------------------------

API_BASE_URL: str = "http://localhost:8000"
DB_PATH: str = "attendance_system.db"

COOLDOWN_SECONDS: int = 180          # 3-minute cooldown per employee
RECOGNITION_EVERY_N: int = 5         # DeepFace is expensive -> run rarely
LIVENESS_EVERY_N: int = 2            # MediaPipe is cheaper -> run more often
FRAME_SLEEP_SECONDS: float = 0.03    # small yield so the loop doesn't peg a core

LATE_CUTOFF_HOUR: int = 9
LATE_CUTOFF_MINUTE: int = 30


# ---------------------------------------------------------------------------
# Session-state / auth helpers
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Ensure every session_state key this app relies on exists."""
    # --- Auth ------------------------------------------------------
    st.session_state.setdefault("jwt_token", None)
    st.session_state.setdefault("role", None)          # "admin" | "employee"
    st.session_state.setdefault("employee_id", None)
    st.session_state.setdefault("employee_name", None)

    # --- Kiosk / Group Recognition ----------------------------------
    st.session_state.setdefault("camera_run", False)
    st.session_state.setdefault("known_embeddings", {})  # {employee_id: [float,...]}
    st.session_state.setdefault("employees_meta", {})    # {employee_id: {"name","department"}}
    st.session_state.setdefault("last_event_message", "")
    st.session_state.setdefault("cooldown_dict", {})      # {employee_id: datetime_last_logged}
    st.session_state.setdefault("latest_faces", [])       # list[dict] from recognize_multiple_faces
    st.session_state.setdefault("latest_liveness", {})    # {employee_id: bool}
    st.session_state.setdefault("event_queue", queue.Queue())


def auth_headers() -> Dict[str, str]:
    """Build an ``Authorization: Bearer <token>`` header dict for the current session."""
    token = st.session_state.get("jwt_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _logout() -> None:
    """Clear every auth-related session_state key, sending the user back to Login."""
    for key in ("jwt_token", "role", "employee_id", "employee_name"):
        st.session_state[key] = None


def authenticated_request(
    method: str, path: str, timeout: float = 8.0, **kwargs
) -> Optional[requests.Response]:
    """``requests.request`` wrapper that attaches the JWT and handles auth failure centrally.

    Every authenticated page in this app should go through this instead of
    calling ``requests.*`` directly, so token attachment and expired-session
    handling only need to be written once.

    Args:
        method: HTTP method, e.g. "GET", "POST", "PUT".
        path: API path starting with "/", appended to ``API_BASE_URL``.
        timeout: Request timeout in seconds.
        **kwargs: Passed straight through to ``requests.request`` (e.g. ``json=...``).

    Returns:
        The ``requests.Response`` on success, or ``None`` if the backend
        couldn't be reached at all, or the token was rejected (401) --
        in the 401 case, this also logs the user out and reruns the app
        so they land back on the login screen.
    """
    headers = kwargs.pop("headers", {})
    headers.update(auth_headers())

    try:
        resp = requests.request(
            method, f"{API_BASE_URL}{path}", headers=headers, timeout=timeout, **kwargs
        )
    except requests.exceptions.RequestException as exc:
        st.warning(f"Couldn't reach the backend at `{API_BASE_URL}`: {exc}")
        return None

    if resp.status_code == 401:
        st.warning("Your session has expired. Please log in again.")
        _logout()
        st.rerun()
        return None

    return resp


# ---------------------------------------------------------------------------
# Login screen
# ---------------------------------------------------------------------------

def login_page() -> None:
    """The app's entry point when nobody is logged in yet (POST /token)."""
    st.set_page_config(page_title="Attendance System -- Login", page_icon="🔐", layout="centered")
    st.title("🔐 Office Attendance System")
    st.caption("Sign in with your employee name and password.")

    with st.form("login_form"):
        username = st.text_input("Name")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In", use_container_width=True)

    if not submitted:
        return

    if not username or not password:
        st.error("Please enter both your name and password.")
        return

    try:
        # NOTE: FastAPI's OAuth2PasswordRequestForm expects
        # application/x-www-form-urlencoded, so this uses `data=` (form
        # encoding), NOT `json=` -- a JSON body here would 422.
        resp = requests.post(
            f"{API_BASE_URL}/token",
            data={"username": username, "password": password},
            timeout=8,
        )
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach the backend at `{API_BASE_URL}`: {exc}")
        return

    if resp.status_code == 401:
        st.error("Incorrect name or password.")
        return
    if resp.status_code != 200:
        st.error(f"Login failed (status {resp.status_code}).")
        return

    payload = resp.json()
    st.session_state.jwt_token = payload["access_token"]
    st.session_state.role = payload["role"]
    st.session_state.employee_id = payload["employee_id"]
    st.session_state.employee_name = payload["name"]
    st.rerun()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def fetch_known_employees() -> None:
    """Pull enrolled employees + embeddings from the backend into session_state.

    Populates ``known_embeddings`` (used directly by ``recognize_multiple_faces``)
    and ``employees_meta`` (used for display names). Admin-only on the
    backend (GET /employees/ requires ``require_admin``), so this is only
    ever called from admin-only pages.
    """
    resp = authenticated_request("GET", "/employees/")
    if resp is None:
        return
    if resp.status_code != 200:
        st.warning(f"Could not load employees (status {resp.status_code}).")
        return

    employees = resp.json()
    embeddings: Dict = {}
    meta: Dict = {}
    for emp in employees:
        emp_id = emp.get("id")
        encoding = emp.get("face_encoding")
        if isinstance(encoding, str):
            try:
                encoding = json.loads(encoding)
            except json.JSONDecodeError:
                continue
        if emp_id is None or not encoding:
            continue
        embeddings[emp_id] = encoding
        meta[emp_id] = {
            "name": emp.get("name", f"Employee {emp_id}"),
            "department": emp.get("department", ""),
        }

    st.session_state.known_embeddings = embeddings
    st.session_state.employees_meta = meta


def compute_status(now: datetime | None = None) -> str:
    """Classify a check-in as 'Present' or 'Late' against a fixed cutoff.

    Args:
        now: Timestamp to classify. Defaults to the current time.

    Returns:
        ``"Present"`` if ``now`` is at/before the cutoff, else ``"Late"``.
    """
    now = now or datetime.now()
    cutoff = now.replace(
        hour=LATE_CUTOFF_HOUR, minute=LATE_CUTOFF_MINUTE, second=0, microsecond=0
    )
    return "Present" if now <= cutoff else "Late"


def send_attendance_log_async(
    employee_id: Union[int, str],
    name: str,
    status: str,
    confidence: float,
    result_queue: "queue.Queue[str]",
) -> None:
    """Background-thread target: POST one attendance log without blocking the UI.

    This is the piece that keeps Group Recognition from freezing the video
    feed. If 4 people are recognized in the same frame, calling
    ``requests.post()`` 4 times *synchronously* in the camera loop would
    stall frame rendering for as long as all 4 requests take combined.
    Instead, each log is dispatched as its own
    ``threading.Thread(target=send_attendance_log_async, ...)``, so the
    requests happen concurrently, off the main thread, while the video
    feed keeps rendering at full speed.

    Because this function runs on a background thread (not Streamlit's
    script-execution thread), it must NEVER call ``st.*`` functions or
    mutate ``st.session_state`` directly. Instead, it puts a plain string
    onto ``result_queue`` (a ``queue.Queue``, thread-safe by design), and
    the main camera loop drains that queue each iteration to update the UI.

    Note: this endpoint is intentionally called WITHOUT an auth header --
    see the module docstring / main.py's security note on why
    /attendance/log has no auth dependency.

    Args:
        employee_id: The recognized employee's id.
        name: Display name, already resolved on the main thread.
        status: "Present" or "Late", already computed on the main thread.
        confidence: Recognition confidence, for the log message.
        result_queue: Thread-safe queue shared with the main loop.
    """
    try:
        resp = requests.post(
            f"{API_BASE_URL}/attendance/log",
            json={"employee_id": employee_id, "status": status},
            timeout=5,
        )
        resp.raise_for_status()
        result_queue.put(
            f"✅ Logged **{name}** as **{status}** ({confidence:.0%} confidence) at {datetime.now():%H:%M:%S}"
        )
    except requests.exceptions.RequestException as exc:
        result_queue.put(f"⚠️ Could not log attendance for {name}: {exc}")


# ---------------------------------------------------------------------------
# Admin Page 1: Live Camera Feed (Attendance Kiosk)
# ---------------------------------------------------------------------------

def _draw_face_box(frame, bbox: tuple, color: tuple, label: str) -> None:
    """Draw one bounding box + label onto a frame in place.

    Args:
        frame: The BGR frame to draw onto (mutated in place).
        bbox: ``(x, y, w, h)`` in pixel coordinates.
        color: BGR color tuple for both the box and its label text.
        label: Text drawn just above the box.
    """
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text_y = max(20, y - 10)
    cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def live_camera_page() -> None:
    """Webcam-driven Group Recognition kiosk: many faces, one video loop.

    For every person visible in frame, this draws a bounding box labeled
    with their name and current status ("Verifying...", "Logging...",
    "Logged!", or "Unknown"), and dispatches attendance logging on a
    background thread per person so a crowd in frame never stalls the
    video feed.
    """
    st.title("📷 Attendance Kiosk -- Group Recognition")
    st.caption("Everyone in frame is tracked at once. Blink naturally to confirm liveness.")

    ctrl_col, metric_col1, metric_col2, metric_col3 = st.columns([1.2, 1, 1, 1])
    with ctrl_col:
        bypass_liveness = st.checkbox("Bypass Liveness (Skip Blink Check)", value=False)
        run = st.checkbox("Start Camera", key="camera_run")
        if st.button("🔄 Refresh Employee List"):
            fetch_known_employees()
            st.success(f"Loaded {len(st.session_state.known_embeddings)} employee(s).")

    if not st.session_state.known_embeddings:
        fetch_known_employees()

    metric_col1.metric("Employees Enrolled", len(st.session_state.known_embeddings))
    metric_col2.metric("Cooldown Window", f"{COOLDOWN_SECONDS // 60} min")
    metric_col3.metric("Faces in Frame", len(st.session_state.latest_faces))

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    if not run:
        frame_placeholder.info("Camera is off. Toggle **Start Camera** to begin.")
        return

    if not st.session_state.known_embeddings:
        st.warning(
            "No enrolled employees are loaded -- everyone will show as "
            "'Unknown'. Register employees on the **Employee Management** page first."
        )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access the webcam. Check camera permissions / device index.")
        return

    # NOTE on non-blocking behavior: this loop relies on Streamlit's own
    # rerun-interrupt mechanism. Every Streamlit API call inside the loop
    # (e.g. frame_placeholder.image(...)) is a checkpoint where Streamlit
    # can safely interrupt and restart the script if the user has changed
    # a widget (e.g. unchecked "Start Camera"). That's why we re-check
    # st.session_state.camera_run every iteration instead of a local
    # variable captured once at the top of the page function.
    skipper = FrameSkipper(
        recognition_every_n=RECOGNITION_EVERY_N,
        liveness_every_n=LIVENESS_EVERY_N,
    )
    frame_count = 0
    event_queue: "queue.Queue[str]" = st.session_state.event_queue

    try:
        while st.session_state.camera_run:
            ok, frame = cap.read()
            if not ok:
                st.error("Lost connection to the webcam.")
                break
            frame_count += 1
            now = datetime.now()

            # -- Recognition: detect + identify every face in the frame --
            if skipper.should_run_recognition(frame_count) and st.session_state.known_embeddings:
                st.session_state.latest_faces = recognize_multiple_faces(
                    frame, st.session_state.known_embeddings
                )
            faces = st.session_state.latest_faces

            # -- Liveness: one MediaPipe pass covering every tracked face --
            if bypass_liveness:
                liveness_by_id = {f["employee_id"]: True for f in faces}
                st.session_state.latest_liveness = liveness_by_id
            else:
                if skipper.should_run_liveness(frame_count) and faces:
                    tracked = [(f["employee_id"], f["bounding_box"]) for f in faces]
                    st.session_state.latest_liveness = detect_liveness_multi(frame, tracked)
                liveness_by_id = st.session_state.latest_liveness

            display_frame = frame.copy()

            for face in faces:
                # Defensive per-face try/except: one malformed detection
                # must not take down box-drawing/logging for everyone else.
                try:
                    employee_id = face["employee_id"]
                    confidence = face["confidence"]
                    bbox = face["bounding_box"]

                    if employee_id == "Unknown":
                        _draw_face_box(display_frame, bbox, (0, 0, 255), "Unknown")
                        continue

                    name = st.session_state.employees_meta.get(employee_id, {}).get(
                        "name", str(employee_id)
                    )
                    is_live = liveness_by_id.get(employee_id, False)

                    # --- Cooldown check (the anti-spam mechanism) -----------
                    # cooldown_dict = {employee_id: last_logged_datetime}.
                    # Checked synchronously, on the main thread, BEFORE we
                    # ever spawn a logging thread for this person -- if we
                    # instead waited for the background POST to finish
                    # before recording the cooldown, every recognition
                    # cycle while the request is still in flight would
                    # spawn another thread for the same still-in-frame
                    # person. Writing cooldown_dict[employee_id] = now
                    # immediately, right before dispatching, closes that race.
                    last_logged_at = st.session_state.cooldown_dict.get(employee_id)
                    in_cooldown = (
                        last_logged_at is not None
                        and (now - last_logged_at).total_seconds() < COOLDOWN_SECONDS
                    )

                    if in_cooldown:
                        _draw_face_box(display_frame, bbox, (0, 200, 0), f"{name} | Logged!")
                    elif not is_live:
                        _draw_face_box(display_frame, bbox, (0, 165, 255), f"{name} | Verifying...")
                    else:
                        # Live + not in cooldown -> log this person now.
                        _draw_face_box(display_frame, bbox, (0, 200, 0), f"{name} | Logging...")

                        st.session_state.cooldown_dict[employee_id] = now
                        status = compute_status(now)

                        # --- Non-blocking dispatch ---------------------------
                        # threading.Thread hands the actual requests.post()
                        # off to a background OS thread and returns
                        # immediately. If N people are recognized in the
                        # same frame, N threads fire concurrently instead
                        # of N sequential blocking calls.
                        threading.Thread(
                            target=send_attendance_log_async,
                            args=(employee_id, name, status, confidence, event_queue),
                            daemon=True,
                        ).start()

                        reset_multi_liveness_state(employee_id)
                except (KeyError, TypeError, ValueError) as exc:
                    logging.getLogger("dashboard").warning(
                        "Skipping one face this frame due to a display error: %s", exc
                    )
                    continue

            frame_placeholder.image(display_frame, channels="BGR")

            # --- Drain the background-thread result queue ---------------
            while not event_queue.empty():
                st.session_state.last_event_message = event_queue.get_nowait()
            if st.session_state.last_event_message:
                status_placeholder.info(st.session_state.last_event_message)

            time.sleep(FRAME_SLEEP_SECONDS)
    finally:
        cap.release()
        reset_multi_liveness_state()


# ---------------------------------------------------------------------------
# Admin Page 2: Employee Management (Registration)
# ---------------------------------------------------------------------------

def employee_management_page() -> None:
    """Registration form: photo -> embedding -> POST /employees/ (admin-only)."""
    st.title("🧑‍💼 Employee Management")
    st.caption("Enroll a new employee's face and set their login credentials.")

    with st.form("registration_form", clear_on_submit=True):
        name = st.text_input("Full Name")
        department = st.text_input("Department")
        password = st.text_input(
            "Initial Password", type="password",
            help="The employee will use this (with their name) to log in to the web/mobile app.",
        )
        is_admin = st.checkbox("Grant Admin Access", value=False)
        photo = st.file_uploader("Employee Photo", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Register Employee")

    if submitted:
        if not name or not department or not password or photo is None:
            st.error("Please fill in every field (name, department, password, photo) before submitting.")
        else:
            _handle_registration(name, department, password, is_admin, photo)

    st.divider()
    st.subheader("Currently Enrolled Employees")

    col_a, _ = st.columns([1, 4])
    with col_a:
        if st.button("Load / Refresh List"):
            fetch_known_employees()

    if st.session_state.employees_meta:
        df = pd.DataFrame(
            [
                {"ID": emp_id, "Name": meta["name"], "Department": meta["department"]}
                for emp_id, meta in st.session_state.employees_meta.items()
            ]
        )
        st.dataframe(df, use_container_width=True)
        with st.expander("Show raw embedding dimensions"):
            dims = {
                emp_id: len(vec) for emp_id, vec in st.session_state.known_embeddings.items()
            }
            st.json(dims)
    else:
        st.info("No employees loaded yet. Click **Load / Refresh List** above.")


def _handle_registration(
    name: str, department: str, password: str, is_admin: bool, photo
) -> None:
    """Extract an embedding from the uploaded photo and POST it to the backend.

    Args:
        name: Employee full name from the form.
        department: Employee department from the form.
        password: Initial login password set by the admin.
        is_admin: Whether to register this employee with the "admin" role.
        photo: A Streamlit ``UploadedFile`` from ``st.file_uploader``.
    """
    tmp_path = None
    embedding = None

    with st.spinner("Extracting facial embedding..."):
        try:
            suffix = os.path.splitext(photo.name)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(photo.getbuffer())
                tmp_path = tmp_file.name

            embedding = register_face(tmp_path)
        except NoFaceDetectedError:
            st.error(
                "No face could be detected in the uploaded photo. "
                "Please use a clear, front-facing, well-lit photo."
            )
        except InvalidImageError:
            st.error("The uploaded file could not be read as an image.")
        except Exception as exc:  # noqa: BLE001 -- surface any unexpected CV/model error
            st.error(f"Unexpected error while extracting the face embedding: {exc}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    if embedding is None:
        return

    resp = authenticated_request(
        "POST", "/employees/",
        json={
            "name": name,
            "department": department,
            "password": password,
            "face_encoding": embedding,
            "role": "admin" if is_admin else "employee",
        },
        timeout=10,
    )
    if resp is None:
        return
    if resp.status_code == 201:
        st.success(f"✅ {name} registered successfully!")
        fetch_known_employees()  # refresh cache so the kiosk sees them immediately
    else:
        detail = resp.json().get("detail", resp.text) if resp.content else resp.status_code
        st.error(f"⚠️ Could not register employee: {detail}")


# ---------------------------------------------------------------------------
# Admin Page 3: Analytics & Reports
# ---------------------------------------------------------------------------

def analytics_page() -> None:
    """Read attendance data straight from SQLite and render charts + table."""
    st.title("📊 Analytics & Reports")
    st.caption(f"Reading directly from local database: `{DB_PATH}`")

    if not os.path.exists(DB_PATH):
        st.warning(
            f"Database file `{DB_PATH}` was not found in the current working "
            f"directory. Make sure the backend has created it and logged at "
            f"least one attendance event."
        )
        return

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        df_logs = pd.read_sql_query("SELECT * FROM attendance_logs", conn)
        df_employees = pd.read_sql_query("SELECT * FROM employees", conn)
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        st.error(f"Could not read attendance data from `{DB_PATH}`: {exc}")
        return
    finally:
        if conn is not None:
            conn.close()

    if df_logs.empty:
        st.info("No attendance logs recorded yet.")
        return

    df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
    merged = df_logs.merge(
        df_employees[["id", "name", "department"]],
        left_on="employee_id",
        right_on="id",
        how="left",
        suffixes=("", "_emp"),
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Logs", len(merged))
    m2.metric("Unique Employees", int(merged["employee_id"].nunique()))
    today_count = int((merged["timestamp"].dt.date == datetime.now().date()).sum())
    m3.metric("Logged Today", today_count)

    st.divider()
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Attendance -- Last 7 Days")
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent = merged[merged["timestamp"] >= seven_days_ago].copy()
        if recent.empty:
            st.info("No attendance logs in the last 7 days.")
        else:
            recent["date"] = recent["timestamp"].dt.date.astype(str)
            daily_counts = recent.groupby("date").size().reset_index(name="count")
            fig_bar = px.bar(
                daily_counts, x="date", y="count",
                title="Daily Check-in Count",
                labels={"count": "Check-ins", "date": "Date"},
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        st.subheader("Status Breakdown")
        if "status" not in merged.columns or merged["status"].dropna().empty:
            st.info("No status data available.")
        else:
            status_counts = merged["status"].value_counts().reset_index()
            status_counts.columns = ["status", "count"]
            fig_pie = px.pie(
                status_counts, names="status", values="count",
                title="Present vs Late",
                color="status",
                color_discrete_map={"Present": "#22c55e", "Late": "#ef4444"},
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()
    st.subheader("Raw Attendance Logs")
    display_cols = [c for c in ["timestamp", "name", "department", "status"] if c in merged.columns]
    st.dataframe(
        merged[display_cols].sort_values("timestamp", ascending=False),
        use_container_width=True,
    )

    with st.expander("Show all raw columns (debug view)"):
        st.dataframe(merged, use_container_width=True)


# ---------------------------------------------------------------------------
# Admin Page 4: Leave Dashboard
# ---------------------------------------------------------------------------

def _decide_leave(leave_id: int, decision: str) -> None:
    """PUT /leaves/{id} with an Approved/Rejected decision, then refresh the page."""
    resp = authenticated_request("PUT", f"/leaves/{leave_id}", json={"status": decision})
    if resp is None:
        return
    if resp.status_code == 200:
        st.success(f"Leave request {decision.lower()}.")
        st.rerun()
    else:
        st.error(f"Could not update leave request (status {resp.status_code}).")


def leave_dashboard_page() -> None:
    """Admin view: approve/reject pending leave, and a calendar of approved leave.

    The "visual calendar" requirement is met with a Plotly timeline
    (Gantt-style bar per approved leave, grouped by employee) rather than
    the third-party ``streamlit-calendar`` package -- Plotly is already a
    hard dependency of this app, so this avoids adding a new one. Swap in
    ``streamlit-calendar`` here if you specifically want a literal
    month-grid calendar widget instead.
    """
    st.title("🗓️ Leave Dashboard")
    st.caption("Review pending leave requests and see approved leave on a timeline.")

    resp = authenticated_request("GET", "/leaves/")
    if resp is None:
        return
    if resp.status_code != 200:
        st.error(f"Could not load leave requests (status {resp.status_code}).")
        return

    leaves = resp.json()
    if not leaves:
        st.info("No leave requests have been submitted yet.")
        return

    df = pd.DataFrame(leaves)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])

    pending = df[df["status"] == "Pending"]
    approved = df[df["status"] == "Approved"]

    st.subheader(f"Pending Requests ({len(pending)})")
    if pending.empty:
        st.success("No pending leave requests. 🎉")
    else:
        for _, row in pending.iterrows():
            with st.container(border=True):
                cols = st.columns([3, 2, 2, 1, 1])
                cols[0].markdown(f"**{row['employee_name']}**")
                cols[1].write(row["start_date"].strftime("%Y-%m-%d"))
                cols[2].write(row["end_date"].strftime("%Y-%m-%d"))
                if cols[3].button("✅ Approve", key=f"approve_{row['id']}"):
                    _decide_leave(int(row["id"]), "Approved")
                if cols[4].button("❌ Reject", key=f"reject_{row['id']}"):
                    _decide_leave(int(row["id"]), "Rejected")
                if row.get("reason"):
                    st.caption(f"Reason: {row['reason']}")

    st.divider()
    st.subheader("Approved Leave Timeline")
    if approved.empty:
        st.info("No approved leave yet.")
    else:
        fig = px.timeline(
            approved, x_start="start_date", x_end="end_date", y="employee_name",
            color="employee_name", title="Approved Leave by Employee",
        )
        fig.update_yaxes(title="Employee")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show all leave requests (including rejected)"):
        st.dataframe(
            df[["employee_name", "start_date", "end_date", "status", "reason"]],
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Employee Page 1: My Dashboard
# ---------------------------------------------------------------------------

def employee_home_page() -> None:
    """Personal attendance summary for the logged-in employee."""
    st.title(f"👋 Welcome, {st.session_state.employee_name}")
    st.caption("Your personal attendance summary.")

    resp = authenticated_request("GET", "/attendance/stats/me")
    if resp is None:
        return
    if resp.status_code != 200:
        st.error(f"Could not load your attendance stats (status {resp.status_code}).")
        return

    stats = resp.json()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Days Logged", stats["total_days_logged"])
    col2.metric("Late Days", stats["late_count"])
    col3.metric("Attendance Rate", f"{stats['attendance_rate']:.0%}")

    if stats["total_days_logged"] > 0:
        pie_df = pd.DataFrame({
            "status": ["Present", "Late"],
            "count": [stats["present_count"], stats["late_count"]],
        })
        fig = px.pie(
            pie_df, names="status", values="count", title="Your Attendance Breakdown",
            color="status", color_discrete_map={"Present": "#22c55e", "Late": "#ef4444"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No attendance logs yet -- check in at the kiosk to get started.")


# ---------------------------------------------------------------------------
# Employee Page 2: Apply for Leave
# ---------------------------------------------------------------------------

def apply_leave_page() -> None:
    """Leave application form + the employee's own leave history."""
    st.title("📝 Apply for Leave")

    with st.form("leave_form", clear_on_submit=True):
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
        reason = st.text_area("Reason (optional)")
        submitted = st.form_submit_button("Submit Request")

    if submitted:
        if end_date < start_date:
            st.error("End date cannot be before the start date.")
        else:
            payload = {
                "start_date": datetime.combine(start_date, datetime.min.time()).isoformat(),
                "end_date": datetime.combine(end_date, datetime.min.time()).isoformat(),
                "reason": reason or None,
            }
            resp = authenticated_request("POST", "/leaves/", json=payload)
            if resp is not None and resp.status_code == 201:
                st.success("Leave request submitted! Awaiting admin approval.")
            elif resp is not None:
                st.error(f"Could not submit leave request (status {resp.status_code}).")

    st.divider()
    st.subheader("Your Leave History")

    resp = authenticated_request("GET", "/leaves/me")
    if resp is None:
        return
    if resp.status_code != 200:
        st.error(f"Could not load your leave history (status {resp.status_code}).")
        return

    my_leaves = resp.json()
    if not my_leaves:
        st.info("You haven't submitted any leave requests yet.")
        return

    df = pd.DataFrame(my_leaves)
    df["start_date"] = pd.to_datetime(df["start_date"]).dt.strftime("%Y-%m-%d")
    df["end_date"] = pd.to_datetime(df["end_date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(df[["start_date", "end_date", "status", "reason"]], use_container_width=True)


# ---------------------------------------------------------------------------
# App entrypoint -- login gate + role-based st.navigation / st.Page
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session_state()

    if not st.session_state.jwt_token:
        login_page()
        return

    st.set_page_config(page_title="Attendance System", page_icon="🕒", layout="wide")

    with st.sidebar:
        st.markdown(f"**{st.session_state.employee_name}**")
        st.caption(f"Role: {st.session_state.role}")
        if st.button("Log out", use_container_width=True):
            _logout()
            st.rerun()
        st.divider()

    if st.session_state.role == "admin":
        pages = [
            st.Page(live_camera_page, title="Live Camera Feed", icon="📷", url_path="kiosk"),
            st.Page(employee_management_page, title="Employee Management", icon="🧑‍💼", url_path="employees"),
            st.Page(analytics_page, title="Analytics & Reports", icon="📊", url_path="analytics"),
            st.Page(leave_dashboard_page, title="Leave Dashboard", icon="🗓️", url_path="leaves"),
        ]
    else:
        pages = [
            st.Page(employee_home_page, title="My Dashboard", icon="🏠", url_path="home"),
            st.Page(apply_leave_page, title="Apply for Leave", icon="📝", url_path="leave"),
        ]

    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
