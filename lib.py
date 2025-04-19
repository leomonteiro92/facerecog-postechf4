import cv2
from tqdm import tqdm
from deepface import DeepFace

# BACKEND_MODEL = "opencv"
# BACKEND_MODEL = "mtcnn"
BACKEND_MODEL = "retinaface"


def load_video(input_path: str, output_path: str):
    """
    1. Load a video file and process it frame by frame."
    """
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in tqdm(range(frames_total), desc="Processing video..."):
        ret, frame = cap.read()

        if not ret:
            break  # End of the video

        # Check if frame has faces or expressions
        frame = detect_face_and_emotion(frame)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def detect_face_and_emotion(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection = DeepFace.analyze(
        frame,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend=BACKEND_MODEL,  # make the code very slow
    )

    for face in detection:
        x, y, w, h = (
            face["region"]["x"],
            face["region"]["y"],
            face["region"]["w"],
            face["region"]["h"],
        )

        dominant_emotion = face["dominant_emotion"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            dominant_emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (36, 255, 12),
            2,
        )

    return frame
