"""
system_utilities.py
====================

Phase 4 (System Integration) utilities for the enterprise Office
Attendance Management System:

    1. ``IntruderManager``       -- stateful consecutive-"Unknown" tracker
                                     that snapshots + reports suspected
                                     intruders, with a cooldown so a single
                                     person standing at the kiosk doesn't
                                     spam the backend.
    2. ``export_monthly_report`` -- pulls attendance + leave data straight
                                     out of SQLite and writes a two-sheet
                                     .xlsx report for the admin team.

IMPORTANT -- assumed backend contract
--------------------------------------
As with ``dashboard.py``, this module is written against a Phase 4
backend surface that doesn't exist in the uploaded ``main.py`` /
``models.py`` / ``schemas.py`` yet (those currently only implement the
Todo API). Specifically, this module assumes:

    POST http://127.0.0.1:8000/intruders/log
        body (matches an ``IntruderCreate`` Pydantic schema):
            {"image_path": str}
        -> 201 Created

    SQLite tables (in ``attendance_system.db``):
        employees(id, name, department, face_encoding, created_at)
        attendance_logs(id, employee_id, timestamp, status)
        leave_requests(id, employee_id, start_date, end_date, status, reason)

Every network call and DB query below is wrapped in error handling and
will raise/log something readable rather than crash your camera loop if
these endpoints/tables aren't there yet -- but you'll need to add them
for either feature to actually do anything.

Dependencies
------------
    pip install opencv-python requests pandas openpyxl
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from datetime import datetime
from typing import Optional, Union

import cv2
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("system_utilities")
logging.basicConfig(level=logging.INFO)


# ===========================================================================
# Feature 1: Intruder Alert Manager
# ===========================================================================

class IntruderManager:
    """Tracks consecutive "Unknown" recognitions and raises throttled alerts.

    Meant to be instantiated once per camera session (not once per frame)
    and fed the result of every recognition cycle via ``update()``. It is
    *not* thread-safe by design -- it assumes a single camera loop is
    calling it sequentially, matching how ``vision_engine.recognize_face``
    is used in ``dashboard.py``.

    Behavior:
        * A running counter increments every time ``update()`` is called
          with ``employee_id == "Unknown"``, and resets to zero the moment
          a *known* employee_id is seen.
        * Once the counter reaches ``unknown_threshold`` consecutive
          "Unknown" frames, an alert fires: the current frame (or a
          tight face crop, if one can be found) is saved to
          ``save_dir`` with a timestamped filename, and a POST request
          is sent to ``api_url`` with that image path.
        * After firing, further alerts are suppressed for
          ``cooldown_seconds`` even if the person is still standing there
          and still "Unknown" -- this is what stops the same intruder
          from generating dozens of alerts per minute.

    Attributes:
        unknown_threshold: Consecutive "Unknown" frames required to fire.
        cooldown_seconds: Minimum seconds between two fired alerts.
        save_dir: Local directory where intruder snapshots are written.
        api_url: Backend endpoint that receives the alert payload.

    Example:
        See the "Integration Example" section at the bottom of this file
        for a full ``cv2.VideoCapture`` loop showing exactly how to wire
        this into your kiosk.
    """

    def __init__(
        self,
        unknown_threshold: int = 5,
        cooldown_seconds: int = 60,
        save_dir: str = "./intruders/",
        api_url: str = "http://127.0.0.1:8000/intruders/log",
    ) -> None:
        """
        Args:
            unknown_threshold: Number of consecutive "Unknown" recognitions
                required before an alert fires. Defaults to 5.
            cooldown_seconds: Seconds to wait after firing before another
                alert can fire, even if still "Unknown". Defaults to 60
                (1 minute), per spec.
            save_dir: Directory to save intruder snapshots into. Created
                automatically if it doesn't already exist.
            api_url: Backend URL that intruder alerts are POSTed to.
        """
        self.unknown_threshold = unknown_threshold
        self.cooldown_seconds = cooldown_seconds
        self.save_dir = save_dir
        self.api_url = api_url

        self._consecutive_unknown: int = 0
        self._last_alert_time: Optional[float] = None

        os.makedirs(self.save_dir, exist_ok=True)

        # Loaded lazily/once so we're not re-reading the cascade file off
        # disk on every single alert.
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def update(self, frame: np.ndarray, employee_id: Union[int, str]) -> bool:
        """Feed one recognition result into the tracker.

        Call this once per recognition cycle (i.e. wherever your loop
        calls ``vision_engine.recognize_face``), passing along whatever
        it returned for ``employee_id``.

        Args:
            frame: The current BGR frame from ``cv2.VideoCapture.read()``.
            employee_id: The recognized employee id, or the string
                ``"Unknown"`` if no match was found.

        Returns:
            ``True`` if this call caused a new alert to fire, ``False``
            otherwise (known face, still below threshold, or still in
            cooldown from a previous alert).
        """
        if employee_id != "Unknown":
            self._consecutive_unknown = 0
            return False

        self._consecutive_unknown += 1
        if self._consecutive_unknown < self.unknown_threshold:
            return False

        now = time.monotonic()
        if self._last_alert_time is not None:
            elapsed = now - self._last_alert_time
            if elapsed < self.cooldown_seconds:
                logger.debug(
                    "Intruder still present but cooling down (%.1fs of %ds).",
                    elapsed, self.cooldown_seconds,
                )
                return False

        image_path = self._save_snapshot(frame)
        self._send_alert(image_path)
        self._last_alert_time = now
        return True

    def reset(self) -> None:
        """Clear the consecutive-unknown counter (e.g. on manual dismiss)."""
        self._consecutive_unknown = 0

    # -- internal helpers ---------------------------------------------

    def _save_snapshot(self, frame: np.ndarray) -> str:
        """Save a cropped face (if detectable) or the full frame to disk.

        Args:
            frame: The current BGR frame.

        Returns:
            The filesystem path the snapshot was written to.
        """
        crop = self._extract_face_crop(frame)
        image_to_save = crop if crop is not None else frame

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"intruder_{timestamp}.jpg"
        path = os.path.join(self.save_dir, filename)

        success = cv2.imwrite(path, image_to_save)
        if not success:
            logger.error("Failed to write intruder snapshot to %s", path)
        else:
            logger.info("Saved intruder snapshot to %s", path)
        return path

    def _extract_face_crop(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Best-effort tight crop around the largest detected face.

        Uses OpenCV's bundled Haar cascade for a fast, dependency-free
        crop. Falls back to ``None`` (caller then saves the full frame)
        if the cascade fails to load or no face is found -- this should
        never block or crash the alert pipeline.

        Args:
            frame: The current BGR frame.

        Returns:
            A cropped BGR image of the largest detected face, or ``None``.
        """
        try:
            if self._face_cascade.empty():
                return None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) == 0:
                return None
            # Largest bounding box = closest/most prominent face.
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            pad = int(0.2 * max(w, h))
            y0, y1 = max(0, y - pad), min(frame.shape[0], y + h + pad)
            x0, x1 = max(0, x - pad), min(frame.shape[1], x + w + pad)
            return frame[y0:y1, x0:x1]
        except cv2.error as exc:
            logger.warning("Face-crop step failed, falling back to full frame: %s", exc)
            return None

    def _send_alert(self, image_path: str) -> None:
        """POST the intruder alert to the backend.

        Payload matches the ``IntruderCreate`` schema referenced in the
        Phase 4 backend spec: ``{"image_path": str}``.

        Args:
            image_path: Local path of the saved snapshot.
        """
        try:
            resp = requests.post(
                self.api_url, json={"image_path": image_path}, timeout=5
            )
            resp.raise_for_status()
            logger.info("Intruder alert logged with backend: %s", image_path)
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Could not reach backend to log intruder alert (%s). "
                "Snapshot was still saved locally at %s.", exc, image_path,
            )


# ===========================================================================
# Feature 2: Monthly Data Exporter
# ===========================================================================

def _month_bounds(reference: Optional[datetime] = None) -> tuple[datetime, datetime]:
    """Return the [start, end) datetime bounds of the current month.

    Args:
        reference: The datetime to compute "current month" relative to.
            Defaults to ``datetime.now()``.

    Returns:
        A ``(month_start, next_month_start)`` tuple usable as an
        inclusive/exclusive SQL range filter.
    """
    now = reference or datetime.now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if month_start.month == 12:
        next_month_start = month_start.replace(year=month_start.year + 1, month=1)
    else:
        next_month_start = month_start.replace(month=month_start.month + 1)
    return month_start, next_month_start


def _autofit_columns(worksheet, dataframe: pd.DataFrame) -> None:
    """Roughly auto-size Excel column widths based on content length.

    Args:
        worksheet: The ``openpyxl`` worksheet to adjust.
        dataframe: The DataFrame that was written to that worksheet
            (used to measure content width per column).
    """
    for col_idx, column in enumerate(dataframe.columns, start=1):
        max_content_len = max(
            [len(str(column))] + [len(str(v)) for v in dataframe[column].astype(str)]
        )
        col_letter = worksheet.cell(row=1, column=col_idx).column_letter
        worksheet.column_dimensions[col_letter].width = min(max_content_len + 2, 40)


def export_monthly_report(
    db_path: str = "attendance_system.db",
    output_filename: str = "monthly_report.xlsx",
) -> str:
    """Export the current month's attendance + all leave data to Excel.

    Connects directly to the local SQLite database (no FastAPI round
    trip), joins ``attendance_logs``/``leave_requests`` against
    ``employees`` to resolve human-readable names, and writes a two-sheet
    ``.xlsx`` workbook: ``Attendance_Logs`` (current month only) and
    ``Leave_Requests`` (all records).

    Args:
        db_path: Path to the SQLite database file.
        output_filename: Path/filename the ``.xlsx`` report is written to.

    Returns:
        The ``output_filename`` that was written, for convenience.

    Raises:
        FileNotFoundError: If ``db_path`` doesn't exist.
        RuntimeError: If the expected tables can't be queried, or the
            Excel file can't be written.

    Example:
        >>> path = export_monthly_report()
        >>> print(f"Report ready at {path}")
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path!r}")

    month_start, next_month_start = _month_bounds()

    conn = sqlite3.connect(db_path)
    try:
        attendance_query = """
            SELECT
                al.id,
                al.timestamp,
                al.status,
                e.name AS employee_name,
                e.department
            FROM attendance_logs al
            JOIN employees e ON al.employee_id = e.id
            WHERE al.timestamp >= ? AND al.timestamp < ?
            ORDER BY al.timestamp
        """
        df_attendance = pd.read_sql_query(
            attendance_query,
            conn,
            params=(
                month_start.strftime("%Y-%m-%d %H:%M:%S"),
                next_month_start.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )

        leave_query = """
            SELECT
                lr.id,
                lr.start_date,
                lr.end_date,
                lr.status,
                lr.reason,
                e.name AS employee_name,
                e.department
            FROM leave_requests lr
            JOIN employees e ON lr.employee_id = e.id
            ORDER BY lr.start_date DESC
        """
        df_leave = pd.read_sql_query(leave_query, conn)
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        raise RuntimeError(
            f"Failed to read report data from {db_path!r}: {exc}\n\n"
            f"Expected tables: attendance_logs(id, employee_id, timestamp, status), "
            f"leave_requests(id, employee_id, start_date, end_date, status, reason), "
            f"employees(id, name, department, ...)."
        ) from exc
    finally:
        conn.close()

    # Nicely formatted timestamps, e.g. "2026-07-01 09:14".
    if not df_attendance.empty and "timestamp" in df_attendance.columns:
        df_attendance["timestamp"] = pd.to_datetime(
            df_attendance["timestamp"]
        ).dt.strftime("%Y-%m-%d %H:%M")

    for date_col in ("start_date", "end_date"):
        if not df_leave.empty and date_col in df_leave.columns:
            df_leave[date_col] = pd.to_datetime(df_leave[date_col]).dt.strftime(
                "%Y-%m-%d %H:%M"
            )

    try:
        with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
            df_attendance.to_excel(writer, sheet_name="Attendance_Logs", index=False)
            df_leave.to_excel(writer, sheet_name="Leave_Requests", index=False)
            _autofit_columns(writer.sheets["Attendance_Logs"], df_attendance)
            _autofit_columns(writer.sheets["Leave_Requests"], df_leave)
    except OSError as exc:
        raise RuntimeError(f"Failed to write Excel report to {output_filename!r}: {exc}") from exc

    logger.info(
        "Monthly report written to %s (%d attendance rows, %d leave rows).",
        output_filename, len(df_attendance), len(df_leave),
    )
    return output_filename


# ===========================================================================
# Demo and Integration Entrypoint
# ===========================================================================

if __name__ == "__main__":
    from datetime import datetime, timedelta
    
    db_file = "attendance_system.db"
    report_file = "monthly_report.xlsx"
    
    print("--------------------------------------------------")
    print("🕒 Starting System Integration Utilities Demo...")
    print("--------------------------------------------------")
    
    # 1. Initialize database with mock tables and data if not present
    if not os.path.exists(db_file):
        print(f"Database '{db_file}' not found. Initializing with mock data...")
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Create schema tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                face_encoding TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leave_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id INTEGER NOT NULL,
                start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL,
                reason TEXT
            )
        """)
        
        # Insert demo employees
        cursor.execute("INSERT INTO employees (name, department, face_encoding) VALUES ('Alice Smith', 'Engineering', '[]')")
        cursor.execute("INSERT INTO employees (name, department, face_encoding) VALUES ('Bob Jones', 'Marketing', '[]')")
        
        # Insert demo logs
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        cursor.execute("INSERT INTO attendance_logs (employee_id, timestamp, status) VALUES (1, ?, 'Present')", (now.strftime("%Y-%m-%d %H:%M:%S"),))
        cursor.execute("INSERT INTO attendance_logs (employee_id, timestamp, status) VALUES (2, ?, 'Late')", (yesterday.strftime("%Y-%m-%d %H:%M:%S"),))
        
        # Insert demo leave request
        cursor.execute("INSERT INTO leave_requests (employee_id, start_date, end_date, status, reason) VALUES (1, ?, ?, 'Approved', 'Family Leave')",
                       (yesterday.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")))
        
        conn.commit()
        conn.close()
        print("✅ Mock database initialized with sample records.")
    
    # 2. Export monthly report
    print(f"Generating monthly report from '{db_file}' to '{report_file}'...")
    try:
        export_monthly_report(db_path=db_file, output_filename=report_file)
        print(f"🚀 Success! Monthly report created at: {os.path.abspath(report_file)}")
    except Exception as e:
        print(f"❌ Error generating report: {e}")
    print("--------------------------------------------------")
