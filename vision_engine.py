"""
vision_engine.py
=================

Core AI & Computer Vision Engine for an enterprise Office Attendance
Management System.

This module provides these capabilities used alongside the FastAPI
backend (see ``main.py`` / ``models.py`` / ``schemas.py``) to power a
webcam-based attendance kiosk -- now supporting multiple simultaneous
people in frame ("Group Recognition"):

    1. ``register_face``            -- enroll a new employee from a still image
    2. ``recognize_face``           -- identify a single employee (legacy, single-person)
    3. ``recognize_multiple_faces`` -- identify every person in a frame at once
    4. ``detect_liveness``          -- single-person anti-spoofing check (blink-based)
    5. ``detect_liveness_multi``    -- per-person anti-spoofing check for a group

Design goals
------------
* **Non-blocking real-time performance.** DeepFace embedding extraction is
  the most expensive operation in this pipeline (tens to low hundreds of
  milliseconds per call depending on hardware). It should therefore
  *never* run on every single frame of a ``cv2.VideoCapture`` loop. See
  ``FrameSkipper`` below and the "Integration Pattern" section for the
  recommended frame-skipping / background-thread strategy.
* **In-memory matching.** ``recognize_face`` never touches disk (unlike
  ``DeepFace.find()``); it only compares the current frame's embedding
  against embeddings already loaded into memory (e.g. pulled once from
  the ``face_encoding`` column of the SQL database at kiosk startup).
* **Stateless, serializable embeddings.** Embeddings are always plain
  Python ``list[float]`` so they round-trip cleanly through
  ``schemas.py`` / a JSON column in the database.

Dependencies
------------
    pip install opencv-python deepface mediapipe numpy

Integration Pattern (frame skipping)
-------------------------------------
DeepFace and MediaPipe are both too slow to run at full webcam frame rate.
The recommended pattern inside your ``cv2.VideoCapture`` loop is::

    cap = cv2.VideoCapture(0)
    frame_count = 0
    skipper = FrameSkipper(recognition_every_n=5, liveness_every_n=2)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        # Liveness is relatively cheap (MediaPipe Face Mesh on CPU) ->
        # run more often.
        if skipper.should_run_liveness(frame_count):
            is_live = detect_liveness(frame)

        # Recognition is expensive (DeepFace embedding) -> run rarely.
        if skipper.should_run_recognition(frame_count):
            employee_id, confidence = recognize_face(frame, known_embeddings)

        cv2.imshow("Attendance Kiosk", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

For an even smoother UI, push ``recognize_face`` onto a background thread
(e.g. ``concurrent.futures.ThreadPoolExecutor``) so the UI thread only
ever blocks on ``cv2.imshow`` / ``cv2.waitKey``. A minimal working example
of this pattern is included in the ``if __name__ == "__main__":`` block
at the bottom of this file.

Group Recognition (multiple people per frame)
-----------------------------------------------
For multi-person kiosks, swap ``recognize_face`` / ``detect_liveness``
for ``recognize_multiple_faces`` / ``detect_liveness_multi``::

    if skipper.should_run_recognition(frame_count):
        faces = recognize_multiple_faces(frame, known_embeddings)
        # faces = [{"employee_id": ..., "confidence": ..., "bounding_box": (x,y,w,h)}, ...]

    if skipper.should_run_liveness(frame_count):
        tracked = [(f["employee_id"], f["bounding_box"]) for f in faces]
        liveness_by_id = detect_liveness_multi(frame, tracked)
        # liveness_by_id = {employee_id: True/False, ...}

Both functions detect *every* face in the frame with a single model pass
(one ``DeepFace.represent`` call, one MediaPipe Face Mesh call) rather
than looping and re-running detection per person, which is what keeps
this CPU-efficient as headcount in frame grows. See each function's
docstring for the full contract.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Tuple, Union

import cv2
import numpy as np

try:
    from deepface import DeepFace
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "DeepFace is required. Install with: pip install deepface"
    ) from exc

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "MediaPipe is required. Install with: pip install mediapipe"
    ) from exc


logger = logging.getLogger("vision_engine")
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------
# Custom exceptions
# --------------------------------------------------------------------------

class VisionEngineError(Exception):
    """Base exception for all vision_engine errors."""


class NoFaceDetectedError(VisionEngineError):
    """Raised when no face could be detected in a supplied image/frame."""


class InvalidImageError(VisionEngineError):
    """Raised when an image path is invalid or the file can't be read."""


# --------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------

#: Embedding model. Facenet512 (512-d) and ArcFace (512-d) are both strong
#: choices for attendance-grade accuracy; Facenet512 is generally a good
#: accuracy/speed tradeoff on CPU-only laptops.
MODEL_NAME: str = "Facenet512"

#: Detector backend used during one-off, higher-accuracy registration.
#: RetinaFace is more accurate but slower -- acceptable for a one-time
#: enrollment call that isn't in the real-time hot path.
REGISTRATION_DETECTOR_BACKEND: str = "retinaface"

#: Detector backend used during real-time recognition. OpenCV's built-in
#: detector is much faster than RetinaFace/MTCNN, which matters when this
#: runs multiple times per second inside a video loop.
REALTIME_DETECTOR_BACKEND: str = "opencv"

#: Cosine-distance threshold below which two embeddings are considered the
#: same person. Lower = stricter. 0.40 is a reasonably strict cutoff for
#: Facenet512-style embeddings.
DEFAULT_MATCH_THRESHOLD: float = 0.40

#: Eye-Aspect-Ratio threshold below which an eye is considered "closed".
EAR_BLINK_THRESHOLD: float = 0.24
 
#: Number of consecutive low-EAR frames required to count as a genuine
#: blink (filters out single-frame detection noise/jitter).
EAR_CONSEC_FRAMES: int = 2
 
#: Size (in frames) of the rolling buffer used to decide whether a blink
#: happened "recently enough" to still count as a liveness signal.
LIVENESS_BUFFER_SIZE: int = 30
 
#: Upper bound on how many simultaneous faces group-recognition /
#: group-liveness will track in a single frame. Caps MediaPipe's
#: max_num_faces so a crowded frame can't silently blow up per-frame
#: CPU cost -- tune to your kiosk's expected foot traffic.
MAX_TRACKED_FACES: int = 10
 
#: Max pixel distance (relative to a face's own bounding-box size) allowed
#: when matching a MediaPipe Face Mesh result to a DeepFace-detected
#: bounding box for the same person. Faces detected by the two models
#: rarely land on the *exact* same box, so matching is done by nearest
#: centroid rather than requiring an exact/IoU match.
_LIVENESS_MATCH_MAX_DISTANCE_RATIO: float = 1.5

#: MediaPipe Face Mesh landmark indices for the six points used to compute
#: EAR per eye (mapped from the classic 68-point dlib scheme onto
#: MediaPipe's 468-point mesh).
_LEFT_EYE_IDX: List[int] = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE_IDX: List[int] = [33, 160, 158, 133, 153, 144]


# --------------------------------------------------------------------------
# 1. register_face
# --------------------------------------------------------------------------

def _apply_clahe_preprocessing(bgr_image: np.ndarray) -> np.ndarray:
    """Improve accuracy under poor/uneven lighting via CLAHE.

    Converts the image to grayscale, applies Contrast Limited Adaptive
    Histogram Equalization (CLAHE) to normalize lighting, then converts
    the result back into a 3-channel image so it can be fed into
    DeepFace's preprocessing pipeline (which expects a color-shaped array).

    Args:
        bgr_image: Original image as loaded by ``cv2.imread`` (BGR, H x W x 3).

    Returns:
        A 3-channel, lighting-normalized image of the same shape as the input.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    normalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return normalized_bgr


def register_face(image_path: str) -> List[float]:
    """Enroll a new employee by extracting a facial embedding from a photo.

    Reads an image from disk, normalizes lighting via grayscale + CLAHE,
    and extracts a facial embedding using DeepFace. This is meant to be
    called once per employee during enrollment (not in the hot path of the
    video loop), so it deliberately favors accuracy over raw speed (e.g.
    by using the RetinaFace detector backend).

    Args:
        image_path: Path to a local image file containing exactly one
            clearly visible face (e.g. an employee ID photo).

    Returns:
        A ``list[float]`` embedding vector, ready to be JSON-serialized
        and stored in the ``face_encoding`` column of the employee record.

    Raises:
        InvalidImageError: If the file doesn't exist or can't be decoded.
        NoFaceDetectedError: If no face can be found in the image.

    Example:
        >>> embedding = register_face("employees/john_doe.jpg")
        >>> import json
        >>> face_encoding_json = json.dumps(embedding)  # store this column
    """
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise InvalidImageError(f"Could not read image at path: {image_path!r}")

    preprocessed = _apply_clahe_preprocessing(bgr_image)

    try:
        result = DeepFace.represent(
            img_path=preprocessed,
            model_name=MODEL_NAME,
            detector_backend=REGISTRATION_DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
        )
    except ValueError as exc:
        # DeepFace raises ValueError (typically "Face could not be
        # detected...") when enforce_detection=True and no face is found.
        raise NoFaceDetectedError(
            f"No face detected in image: {image_path!r}"
        ) from exc

    if not result:
        raise NoFaceDetectedError(f"No face detected in image: {image_path!r}")

    # DeepFace.represent returns a list of dicts (one per detected face).
    # Registration assumes a single, well-framed face -- take the first.
    embedding: List[float] = [float(x) for x in result[0]["embedding"]]
    return embedding


# --------------------------------------------------------------------------
# 2. recognize_face
# --------------------------------------------------------------------------

def _extract_embedding_from_frame(
    frame: np.ndarray,
    detector_backend: str = REALTIME_DETECTOR_BACKEND,
) -> Union[List[float], None]:
    """Extract a single facial embedding from a live BGR video frame.

    Args:
        frame: A single frame from ``cv2.VideoCapture.read()`` (BGR).
        detector_backend: DeepFace detector backend to use.

    Returns:
        The embedding as a list of floats, or ``None`` if no face was
        detected. Recognition should degrade gracefully on empty frames
        rather than raising on every frame with nobody in view.
    """
    try:
        result = DeepFace.represent(
            img_path=frame,
            model_name=MODEL_NAME,
            detector_backend=detector_backend,
            enforce_detection=True,
            align=True,
        )
    except ValueError:
        return None

    if not result:
        return None

    return [float(x) for x in result[0]["embedding"]]


def recognize_face(
    frame: np.ndarray,
    known_embeddings: Dict[Union[int, str], List[float]],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> Tuple[Union[int, str], float]:
    """Identify the employee in a live frame against known embeddings.

    Extracts an embedding from the current frame and compares it, entirely
    in memory, against every known employee embedding using cosine
    distance. This intentionally avoids ``DeepFace.find()`` (which
    re-reads a directory of images from disk on every call) in favor of a
    single vectorized NumPy comparison against embeddings already loaded
    into RAM (e.g. cached once at kiosk startup from the database).

    Args:
        frame: A single BGR frame from ``cv2.VideoCapture.read()``.
        known_embeddings: Mapping of ``employee_id -> embedding vector``,
            e.g. loaded once at startup via
            ``{row.id: json.loads(row.face_encoding) for row in employees}``.
        threshold: Maximum cosine distance for a match to be accepted.
            Lower is stricter. Defaults to ``DEFAULT_MATCH_THRESHOLD``.

    Returns:
        A tuple of ``(employee_id, confidence)`` where ``employee_id`` is
        either the matched key from ``known_embeddings`` or the string
        ``"Unknown"``, and ``confidence`` is ``1 - cosine_distance`` for
        the best match found (``0.0`` if no face was detected at all).

    Example:
        >>> employee_id, confidence = recognize_face(frame, known_embeddings)
        >>> if employee_id != "Unknown":
        ...     log_attendance(employee_id, confidence)
    """
    if not known_embeddings:
        logger.warning("recognize_face called with an empty known_embeddings dict.")
        return "Unknown", 0.0

    query_embedding = _extract_embedding_from_frame(frame)
    if query_embedding is None:
        return "Unknown", 0.0

    ids = list(known_embeddings.keys())
    matrix = np.array([known_embeddings[i] for i in ids], dtype=np.float64)
    query = np.array(query_embedding, dtype=np.float64)

    # Vectorized cosine distance across every known embedding at once:
    # distance = 1 - (a . b) / (|a| * |b|)
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    similarities = matrix_norm @ query_norm
    distances = 1.0 - similarities

    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])
    confidence = 1.0 - best_distance

    if best_distance <= threshold:
        return ids[best_idx], confidence
    return "Unknown", confidence


# --------------------------------------------------------------------------
# 2b. recognize_multiple_faces (Group Recognition)
# --------------------------------------------------------------------------

def _extract_all_embeddings_from_frame(
    frame: np.ndarray,
    detector_backend: str = "mediapipe",
) -> List[Dict[str, Any]]:
    """Detect every face in a frame and extract an embedding + bbox for each.

    Uses a single ``DeepFace.represent()`` call with ``enforce_detection=False``
    so the underlying detector scans the whole frame once and returns one
    result per face found, rather than calling ``recognize_face`` in a loop
    (which would redundantly re-run detection from scratch once per
    already-detected face).

    Args:
        frame: A single BGR frame from ``cv2.VideoCapture.read()``.
        detector_backend: DeepFace detector backend to use.

    Returns:
        A list of dicts, one per detected face, each with:
            "embedding": list[float]
            "bounding_box": (x, y, w, h) in pixel coordinates
        Empty list if no faces were found.
    """
    try:
        results = DeepFace.represent(
            img_path=frame,
            model_name=MODEL_NAME,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True,
        )
    except ValueError:
        return []

    faces: List[Dict[str, Any]] = []
    for res in results:
        # Gotcha: with enforce_detection=False, a totally faceless frame
        # still returns one placeholder result spanning the whole image
        # with face_confidence == 0. Filter that out explicitly, or an
        # empty frame gets reported as "one face, the whole picture".
        if res.get("face_confidence", 1.0) == 0:
            continue

        area = res.get("facial_area", {})
        bbox = (
            int(area.get("x", 0)),
            int(area.get("y", 0)),
            int(area.get("w", frame.shape[1])),
            int(area.get("h", frame.shape[0])),
        )
        faces.append({
            "embedding": [float(v) for v in res["embedding"]],
            "bounding_box": bbox,
        })
    return faces


def recognize_multiple_faces(
    frame: np.ndarray,
    known_embeddings: Dict[Union[int, str], List[float]],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    detector_backend: str = "mediapipe",
) -> List[Dict[str, Any]]:
    """Detect and identify every person visible in a single frame.

    This is the Group Recognition counterpart to ``recognize_face``: face
    *detection* runs once for the whole frame (one ``DeepFace.represent``
    call finds all faces + their bounding boxes together), and then each
    detected face's embedding is matched against ``known_embeddings``
    using the same vectorized, in-memory cosine-distance approach as
    ``recognize_face`` -- one matrix multiply against every known
    embedding, run once per *detected face* rather than once per known
    employee. Cost scales as (people in frame) x (enrolled employees),
    same as before, just batched across everyone in the frame instead of
    assuming exactly one person.

    Args:
        frame: A single BGR frame from ``cv2.VideoCapture.read()``.
        known_embeddings: Mapping of ``employee_id -> embedding vector``.
        threshold: Maximum cosine distance for a match to be accepted.

    Returns:
        A list of dicts, one per detected face, each with:
            "employee_id": matched id, or the string "Unknown"
            "confidence": 1 - cosine_distance for the best match
            "bounding_box": (x, y, w, h) in pixel coordinates, for
                drawing a box around this person on screen
        A face that fails embedding/matching for any reason (e.g.
        heavily obscured, extreme angle) is skipped with a logged
        warning rather than raising -- one bad face in a crowd never
        takes down recognition for everyone else in the same frame.

    Example:
        >>> faces = recognize_multiple_faces(frame, known_embeddings)
        >>> for face in faces:
        ...     x, y, w, h = face["bounding_box"]
        ...     label = f'{face["employee_id"]} ({face["confidence"]:.0%})'
        ...     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ...     cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    """
    detected_faces = _extract_all_embeddings_from_frame(frame)
    if not detected_faces:
        return []

    matrix_norm: Union[np.ndarray, None] = None
    ids: List[Union[int, str]] = []
    if known_embeddings:
        ids = list(known_embeddings.keys())
        matrix = np.array([known_embeddings[i] for i in ids], dtype=np.float64)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)

    results: List[Dict[str, Any]] = []
    for face in detected_faces:
        try:
            bbox = face["bounding_box"]

            if matrix_norm is None:
                results.append({"employee_id": "Unknown", "confidence": 0.0, "bounding_box": bbox})
                continue

            query = np.array(face["embedding"], dtype=np.float64)
            query_norm = query / (np.linalg.norm(query) + 1e-10)

            # Same vectorized cosine-distance trick as recognize_face(),
            # just run once per detected face instead of once total.
            similarities = matrix_norm @ query_norm
            distances = 1.0 - similarities
            best_idx = int(np.argmin(distances))
            best_distance = float(distances[best_idx])
            confidence = 1.0 - best_distance

            employee_id = ids[best_idx] if best_distance <= threshold else "Unknown"
            results.append({
                "employee_id": employee_id,
                "confidence": confidence,
                "bounding_box": bbox,
            })
        except (KeyError, ValueError, IndexError) as exc:
            logger.warning("Skipping one face in frame due to a matching error: %s", exc)
            continue

    return results


# --------------------------------------------------------------------------
# 3. detect_liveness (anti-spoofing)
# --------------------------------------------------------------------------

def _eye_aspect_ratio(landmarks: np.ndarray, eye_idx: List[int]) -> float:
    """Compute the Eye Aspect Ratio (EAR) for one eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    A low EAR indicates the eyelid is closed; a dip-and-recovery in EAR
    over a short window of frames is interpreted as a blink.

    Args:
        landmarks: Array of (x, y) pixel coordinates for all face-mesh
            landmarks in the current frame.
        eye_idx: The six landmark indices (in dlib-EAR order) for one eye.

    Returns:
        The EAR value as a float.
    """
    p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in eye_idx)
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-10)


@dataclass
class LivenessDetector:
    """Stateful, blink-based liveness detector.

    A printed photo or a static image held up to the camera cannot easily
    reproduce a natural eye blink on demand, so tracking EAR over a
    rolling window of frames is a cheap and effective anti-spoofing signal
    that runs entirely on-device with MediaPipe (CPU only, no GPU needed).

    This class holds state (a rolling EAR history and blink counter)
    across calls, since liveness cannot be determined from a single frame
    in isolation. Use the module-level ``detect_liveness()`` convenience
    function, which wraps a shared singleton instance of this class, or
    instantiate your own ``LivenessDetector`` per camera/session if you
    need to track multiple independent kiosks concurrently.

    Attributes:
        ear_history: Rolling buffer of recent average EAR values.
        blink_confirmed_buffer: Rolling buffer of booleans indicating
            whether a completed blink was observed at that frame.
        consecutive_low_frames: Count of consecutive frames where EAR was
            below the blink threshold (used to confirm a full blink rather
            than a single noisy frame).
    """

    ear_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=LIVENESS_BUFFER_SIZE)
    )
    blink_confirmed_buffer: Deque[bool] = field(
        default_factory=lambda: deque(maxlen=LIVENESS_BUFFER_SIZE)
    )
    consecutive_low_frames: int = 0

    def __post_init__(self) -> None:
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def update(self, frame: np.ndarray) -> bool:
        """Feed one frame into the detector and return current liveness state.

        Args:
            frame: A single BGR frame from ``cv2.VideoCapture.read()``.

        Returns:
            ``True`` if a blink has been confirmed within the rolling
            buffer window (i.e. the subject is very likely a live person),
            otherwise ``False``.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_frame)

        blink_this_frame = False

        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            mesh = results.multi_face_landmarks[0]
            landmarks = np.array(
                [(lm.x * w, lm.y * h) for lm in mesh.landmark], dtype=np.float64
            )

            left_ear = _eye_aspect_ratio(landmarks, _LEFT_EYE_IDX)
            right_ear = _eye_aspect_ratio(landmarks, _RIGHT_EYE_IDX)
            avg_ear = (left_ear + right_ear) / 2.0
            self.ear_history.append(avg_ear)

            if avg_ear < EAR_BLINK_THRESHOLD:
                self.consecutive_low_frames += 1
            else:
                # Eyes just reopened after being closed for enough frames
                # to be a real blink, not detector jitter.
                if self.consecutive_low_frames >= EAR_CONSEC_FRAMES:
                    blink_this_frame = True
                self.consecutive_low_frames = 0
        else:
            # No face this frame -- don't crash, just record "no blink".
            self.consecutive_low_frames = 0

        self.blink_confirmed_buffer.append(blink_this_frame)
        return any(self.blink_confirmed_buffer)

    def reset(self) -> None:
        """Clear all rolling state (call after each check-in attempt)."""
        self.ear_history.clear()
        self.blink_confirmed_buffer.clear()
        self.consecutive_low_frames = 0

    def close(self) -> None:
        """Release the underlying MediaPipe FaceMesh resources."""
        self._face_mesh.close()


# Module-level singleton so `detect_liveness(frame)` matches the requested
# simple function signature while still maintaining the temporal state a
# blink check inherently requires.
_liveness_detector = LivenessDetector()


def detect_liveness(frame: np.ndarray) -> bool:
    """Anti-spoofing check: confirm a real, living person is in frame.

    Wraps a shared ``LivenessDetector`` instance that tracks Eye Aspect
    Ratio (EAR) over a rolling buffer of recent frames to detect a natural
    blink. A printed photo or a static image held up to the camera cannot
    blink, which stops the overwhelming majority of naive photo-spoofing
    attempts.

    Call this on every frame (or every 2nd-3rd frame -- see
    ``FrameSkipper``) while a person is standing in front of the kiosk.
    Because liveness needs several frames of history, don't expect
    ``True`` on the very first call for a given person; call
    ``reset_liveness_state()`` after each check-in attempt so state
    doesn't leak between different employees.

    Args:
        frame: A single BGR frame from ``cv2.VideoCapture.read()``.

    Returns:
        ``True`` if a blink was confirmed within the rolling buffer
        window, ``False`` otherwise.

    Note:
        For stronger protection against video-replay attacks (e.g. a
        tablet playing a recording of the employee blinking), combine
        this with head-pose variance tracking: sample the face mesh's
        yaw/pitch via the nose-tip and eye-corner landmarks over ~10
        frames and require a small but nonzero amount of natural head
        movement, since a phone/tablet held flat produces a suspiciously
        flat, planar landmark trajectory compared to a real head.
    """
    return _liveness_detector.update(frame)


def reset_liveness_state() -> None:
    """Reset the shared liveness detector's rolling state.

    Call this after each check-in attempt (success or failure) so a blink
    from the previous person in line doesn't count toward the next
    person's liveness check.
    """
    _liveness_detector.reset()


# --------------------------------------------------------------------------
# 3b. detect_liveness_multi (Group Recognition liveness)
# --------------------------------------------------------------------------

@dataclass
class MultiFaceLivenessTracker:
    """Blink-based liveness tracking for several simultaneous faces.

    CPU-efficiency choice: this runs a *single* MediaPipe Face Mesh pass
    per frame (``max_num_faces=MAX_TRACKED_FACES``) rather than cropping
    each detected face and running Face Mesh once per person. MediaPipe's
    own face detector already scans the entire frame once internally
    regardless of how many faces it's configured to track, so one batched
    call is strictly cheaper than N independent single-face calls, each of
    which would re-run detection from scratch on a cropped sub-image.

    Because MediaPipe's per-frame face ordering isn't guaranteed to be
    stable, each returned mesh is matched to the caller-supplied bounding
    boxes (typically the ones ``recognize_multiple_faces`` just produced)
    by nearest centroid distance -- not by list position.

    Every tracked identity gets its own rolling EAR history and blink
    buffer, keyed by whatever ``identity_key`` the caller passes in
    (normally an ``employee_id``). This is what lets liveness be judged
    per-person instead of one shared buffer for the whole frame: person A
    blinking doesn't count as person B having blinked.

    Note:
        Faces still recognized as "Unknown" don't have a stable per-person
        key across frames (two different strangers would otherwise share
        one "Unknown" bucket). That's fine in practice -- liveness only
        needs to be authoritative for faces you're about to log
        attendance for, and those are, by definition, matched to a real
        employee_id.
    """

    ear_histories: Dict[Any, Deque[float]] = field(default_factory=dict)
    blink_buffers: Dict[Any, Deque[bool]] = field(default_factory=dict)
    consecutive_low: Dict[Any, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=MAX_TRACKED_FACES,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def update(
        self,
        frame: np.ndarray,
        tracked_faces: List[Tuple[Any, Tuple[int, int, int, int]]],
    ) -> Dict[Any, bool]:
        """Feed one frame + this frame's detected boxes into the tracker.

        Args:
            frame: A single BGR frame from ``cv2.VideoCapture.read()``.
            tracked_faces: List of ``(identity_key, bounding_box)`` pairs
                for this frame, e.g. built from
                ``recognize_multiple_faces()`` output as::

                    [(f["employee_id"], f["bounding_box"]) for f in faces]

        Returns:
            Dict mapping each ``identity_key`` from ``tracked_faces`` to
            ``True``/``False`` liveness for that person. A face that
            can't be matched to a MediaPipe mesh this frame (occlusion,
            extreme angle) simply doesn't accumulate a blink this frame
            rather than raising -- it does not affect any other tracked
            face's result.
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = self._face_mesh.process(rgb_frame)

        # Pre-compute (centroid, landmarks) for every mesh MediaPipe found
        # this frame, once, so matching each tracked face below is a cheap
        # nearest-neighbor lookup rather than re-running Face Mesh per box.
        mesh_candidates: List[Tuple[np.ndarray, np.ndarray]] = []
        if mp_results.multi_face_landmarks:
            for mesh in mp_results.multi_face_landmarks:
                landmarks = np.array(
                    [(lm.x * w, lm.y * h) for lm in mesh.landmark], dtype=np.float64
                )
                centroid = landmarks.mean(axis=0)
                mesh_candidates.append((centroid, landmarks))

        liveness_by_identity: Dict[Any, bool] = {}

        for identity_key, bbox in tracked_faces:
            self.ear_histories.setdefault(identity_key, deque(maxlen=LIVENESS_BUFFER_SIZE))
            self.blink_buffers.setdefault(identity_key, deque(maxlen=LIVENESS_BUFFER_SIZE))
            self.consecutive_low.setdefault(identity_key, 0)

            try:
                blink_this_frame = False
                matched_landmarks = self._match_landmarks_to_bbox(bbox, mesh_candidates)

                if matched_landmarks is not None:
                    left_ear = _eye_aspect_ratio(matched_landmarks, _LEFT_EYE_IDX)
                    right_ear = _eye_aspect_ratio(matched_landmarks, _RIGHT_EYE_IDX)
                    avg_ear = (left_ear + right_ear) / 2.0
                    self.ear_histories[identity_key].append(avg_ear)

                    if avg_ear < EAR_BLINK_THRESHOLD:
                        self.consecutive_low[identity_key] += 1
                    else:
                        if self.consecutive_low[identity_key] >= EAR_CONSEC_FRAMES:
                            blink_this_frame = True
                        self.consecutive_low[identity_key] = 0
                else:
                    # No matching mesh this frame (occlusion/angle) -- don't
                    # crash, just don't advance this person's blink state.
                    self.consecutive_low[identity_key] = 0

                self.blink_buffers[identity_key].append(blink_this_frame)
                liveness_by_identity[identity_key] = any(self.blink_buffers[identity_key])
            except (ValueError, IndexError, ZeroDivisionError) as exc:
                logger.warning(
                    "Liveness check failed for %r this frame, marking not-live: %s",
                    identity_key, exc,
                )
                liveness_by_identity[identity_key] = False

        return liveness_by_identity

    @staticmethod
    def _match_landmarks_to_bbox(
        bbox: Tuple[int, int, int, int],
        mesh_candidates: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Union[np.ndarray, None]:
        """Find the MediaPipe mesh whose centroid is closest to a bbox's center.

        Args:
            bbox: ``(x, y, w, h)`` bounding box for one detected face.
            mesh_candidates: ``(centroid, landmarks)`` pairs from this
                frame's MediaPipe Face Mesh pass.

        Returns:
            The best-matching landmarks array, or ``None`` if nothing was
            close enough to trust as the same face.
        """
        if not mesh_candidates:
            return None

        x, y, w, h = bbox
        bbox_centroid = np.array([x + w / 2.0, y + h / 2.0])

        distances = [np.linalg.norm(centroid - bbox_centroid) for centroid, _ in mesh_candidates]
        best_idx = int(np.argmin(distances))

        max_allowed_distance = max(w, h) * _LIVENESS_MATCH_MAX_DISTANCE_RATIO
        if distances[best_idx] > max_allowed_distance:
            return None
        return mesh_candidates[best_idx][1]

    def reset(self, identity_key: Any = None) -> None:
        """Clear rolling state for one identity, or every tracked identity.

        Args:
            identity_key: If given, clear only this person's state (e.g.
                right after logging their attendance). If ``None``, clear
                every tracked identity at once (e.g. at kiosk startup).
        """
        if identity_key is None:
            self.ear_histories.clear()
            self.blink_buffers.clear()
            self.consecutive_low.clear()
        else:
            self.ear_histories.pop(identity_key, None)
            self.blink_buffers.pop(identity_key, None)
            self.consecutive_low.pop(identity_key, None)

    def close(self) -> None:
        """Release the underlying MediaPipe FaceMesh resources."""
        self._face_mesh.close()


# Module-level singleton, mirroring the single-face `_liveness_detector`
# pattern above.
_multi_liveness_tracker = MultiFaceLivenessTracker()


def detect_liveness_multi(
    frame: np.ndarray,
    tracked_faces: List[Tuple[Any, Tuple[int, int, int, int]]],
) -> Dict[Any, bool]:
    """Anti-spoofing check for every person recognized in the current frame.

    Group Recognition counterpart to ``detect_liveness``: pass in the
    ``(employee_id, bounding_box)`` pairs you just got back from
    ``recognize_multiple_faces()`` and get back a per-person liveness
    verdict, each backed by its own rolling blink-history buffer so one
    person's blink can't "cover" for someone standing next to them.

    Args:
        frame: A single BGR frame from ``cv2.VideoCapture.read()``.
        tracked_faces: ``[(identity_key, bounding_box), ...]`` for every
            face detected this frame.

    Returns:
        ``{identity_key: True/False, ...}`` liveness for each entry in
        ``tracked_faces``.

    Example:
        >>> faces = recognize_multiple_faces(frame, known_embeddings)
        >>> tracked = [(f["employee_id"], f["bounding_box"]) for f in faces]
        >>> liveness_by_id = detect_liveness_multi(frame, tracked)
        >>> for face in faces:
        ...     if face["employee_id"] != "Unknown" and liveness_by_id[face["employee_id"]]:
        ...         log_attendance(face["employee_id"], face["confidence"])
    """
    return _multi_liveness_tracker.update(frame, tracked_faces)


def reset_multi_liveness_state(identity_key: Any = None) -> None:
    """Reset the shared multi-face tracker's rolling state.

    Args:
        identity_key: Clear just this person's blink history (e.g. right
            after their attendance is logged), or clear everyone's if
            ``None``.
    """
    _multi_liveness_tracker.reset(identity_key)


# --------------------------------------------------------------------------
# Frame-skipping helper for real-time integration
# --------------------------------------------------------------------------

class FrameSkipper:
    """Utility for deciding which expensive operations to run on which frames.

    DeepFace embedding extraction is far more expensive than MediaPipe
    Face Mesh, which is itself more expensive than simply displaying a
    frame. Running every operation on every frame will drop a webcam feed
    from ~30 FPS to a few FPS or less. This helper centralizes the "every
    Nth frame" logic referenced throughout this module's docstrings.

    Example:
        >>> skipper = FrameSkipper(recognition_every_n=5, liveness_every_n=2)
        >>> if skipper.should_run_recognition(frame_count):
        ...     employee_id, confidence = recognize_face(frame, known_embeddings)
    """

    def __init__(self, recognition_every_n: int = 5, liveness_every_n: int = 2) -> None:
        """
        Args:
            recognition_every_n: Run ``recognize_face`` only on every Nth
                frame (default: every 5th frame, ~6 times/sec at 30 FPS).
            liveness_every_n: Run ``detect_liveness`` only on every Nth
                frame (default: every 2nd frame, ~15 times/sec at 30 FPS).
        """
        self.recognition_every_n = max(1, recognition_every_n)
        self.liveness_every_n = max(1, liveness_every_n)

    def should_run_recognition(self, frame_count: int) -> bool:
        """Return True if ``recognize_face`` should run on this frame."""
        return frame_count % self.recognition_every_n == 0

    def should_run_liveness(self, frame_count: int) -> bool:
        """Return True if ``detect_liveness`` should run on this frame."""
        return frame_count % self.liveness_every_n == 0


# --------------------------------------------------------------------------
# Demo integration loop (run this file directly to try the webcam pipeline)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal demo wiring the three functions together with frame-skipping
    # and a background thread for the expensive recognition call, so the
    # UI thread never blocks waiting on DeepFace.
    #
    # In your real FastAPI-backed kiosk app, replace `known_embeddings`
    # with a dict loaded once at startup from the database, e.g.:
    #
    #     db_employees = crud.get_employees(db)
    #     known_embeddings = {
    #         e.id: json.loads(e.face_encoding) for e in db_employees
    #     }
    import sys
    from concurrent.futures import ThreadPoolExecutor

    known_embeddings: Dict[Union[int, str], List[float]] = {}
    # Example: uncomment and point at a real enrollment photo to test.
    # known_embeddings[1] = register_face("employees/1.jpg")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam.")
        sys.exit(1)

    executor = ThreadPoolExecutor(max_workers=1)
    recognition_future = None
    skipper = FrameSkipper(recognition_every_n=5, liveness_every_n=2)
    frame_count = 0
    last_label = "Scanning..."
    live_status = "Checking liveness..."

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            if skipper.should_run_liveness(frame_count):
                is_live = detect_liveness(frame)
                live_status = "LIVE" if is_live else "Checking liveness..."

            if skipper.should_run_recognition(frame_count) and known_embeddings:
                if recognition_future is None or recognition_future.done():
                    if recognition_future is not None:
                        emp_id, conf = recognition_future.result()
                        last_label = f"{emp_id} ({conf:.2f})"
                    recognition_future = executor.submit(
                        recognize_face, frame.copy(), known_embeddings
                    )

            cv2.putText(
                frame, f"{last_label} | {live_status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.imshow("Attendance Kiosk (demo)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        executor.shutdown(wait=False)
        _liveness_detector.close()
        _multi_liveness_tracker.close()
