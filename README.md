👁️ AI-Powered Full-Stack Attendance System

📌 Project Overview

An enterprise-grade, full-stack Office Attendance Management System that replaces traditional biometric scanners with a real-time computer vision pipeline. The system features a decoupled architecture with an asynchronous FastAPI backend, a multi-person face recognition engine, and role-based web dashboards.

Key Highlight: Engineered to handle real-time group recognition with anti-spoofing (liveness detection) using in-memory cosine similarity and non-blocking background threading, ensuring zero UI lag during live video processing.

✨ Core Features

👥 Multi-Person Face Recognition: Scans crowds and detects multiple faces simultaneously using DeepFace (Facenet512/ArcFace) and OpenCV.

🛡️ Anti-Spoofing (Liveness Detection): Uses MediaPipe Face Mesh to detect blinking and head pose variance, preventing intruders from spoofing the system with photos or screens.

🚨 Intruder Alert System: Stateful tracking of unknown faces. Triggers an alert and saves cropped images of unauthorized personnel after 5 consecutive unrecognized frames.

📊 Role-Based Dashboards (Streamlit):

Admin View: Live camera feed (kiosk mode), employee registration, global attendance analytics (Plotly), and a leave management approval system.

Employee View: Personal attendance statistics and leave request forms.

🔐 Secure Authentication: JWT (JSON Web Tokens) and bcrypt password hashing for all API endpoints.

📑 Automated Reporting: Exports monthly attendance and leave data directly to formatted Excel (.xlsx) files.

🛠️ Tech Stack

Backend: Python, FastAPI (Async), Uvicorn, SQLAlchemy 2.0, Pydantic v2.

Database: SQLite (Local Development) -> Ready for PostgreSQL (Production).

Computer Vision: OpenCV, DeepFace, MediaPipe, Numpy, Scipy.

Frontend/UI: Streamlit, Plotly, Pandas.

Security: passlib (bcrypt), python-jose (JWT).

🏗️ System Architecture

The application is highly decoupled. A local webcam client (vision_engine.py) runs the heavy computer vision loops and dispatches lightweight, non-blocking HTTP requests via background threads to the FastAPI backend (main.py). The Streamlit Dashboard (dashboard.py) acts as the presentation layer, reading from the database and managing user roles.

🚀 Installation & Setup

1. Prerequisites

Ensure you have Python 3.10+ installed. A webcam is required for the live recognition features.

2. Clone the Repository

git clone [https://github.com/yourusername/ai-attendance-system.git](https://github.com/yourusername/ai-attendance-system.git)
cd ai-attendance-system


3. Install Dependencies

It is highly recommended to use a virtual environment.

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


(If requirements.txt is missing, install the core packages: fastapi uvicorn sqlalchemy aiosqlite pydantic opencv-python deepface mediapipe streamlit pandas plotly openpyxl passlib[bcrypt] python-jose[cryptography] python-multipart)

4. Running the Application

The system requires both the backend and frontend to be running. You can run them in separate terminal windows:

Terminal 1: Start the Backend (FastAPI)

uvicorn main:app --reload


The API will be available at http://127.0.0.1:8000 (Visit /docs for the interactive Swagger UI).

Terminal 2: Start the Dashboard (Streamlit)

streamlit run dashboard.py


The dashboard will automatically open in your default web browser.

📂 Project Structure

📦 ai-attendance-system
 ┣ 📂 intruders/            # Auto-generated directory for unauthorized face captures
 ┣ 📜 main.py               # FastAPI backend, DB models, and Auth endpoints
 ┣ 📜 vision_engine.py      # Core CV pipeline (DeepFace, MediaPipe, embeddings)
 ┣ 📜 dashboard.py          # Streamlit UI (Live camera, Analytics, HR portal)
 ┣ 📜 system_utilities.py   # Intruder state manager & Excel data exporters
 ┗ 📜 README.md             # Project documentation


🔮 Future Roadmap

[ ] React Native Mobile App: A dedicated mobile application for employees to view personal stats and apply for leave on iOS/Android.

[ ] PostgreSQL Migration: Switch from SQLite to a cloud-hosted PostgreSQL database for multi-kiosk production deployment.

[ ] Two-Factor Auth (2FA): Integrate SIFT (Scale-Invariant Feature Transform) for physical ID card verification alongside facial recognition.

🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
