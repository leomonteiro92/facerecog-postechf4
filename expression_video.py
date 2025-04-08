import os
import cv2
import cv2.data
import face_recognition
import numpy as np
from tqdm import tqdm
from deepface import DeepFace


def index_known_faces(folder_path: str):
    """
    Index known faces from images in a folder.
    """
    known_face_encondings = []
    known_face_names = []

    for filename in os.listdir(folder_path):
        if not filename.endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = os.path.join(folder_path, filename)
        image = face_recognition.load_image_file(image_path)

        face_encodings = face_recognition.face_encodings(image)

        if not face_encodings:
            print(f"No face found in image: {filename}")
            continue

        face_encoding = face_encodings[0]
        name = os.path.splitext(filename)[0][:-1]
        known_face_encondings.append(face_encoding)
        known_face_names.append(name)

    return known_face_encondings, known_face_names


def detect_faces_and_emotions(
    input_path: str,
    output_path: str,
    known_face_encondings,
    known_face_names,
):
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

        detections = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
        )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_enc in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encondings,
                face_enc,
            )
            name = "Unknown"
            face_distances = face_recognition.face_distance(
                known_face_encondings,
                face_enc,
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

        for face in detections:
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
                0.9,
                (36, 255, 12),
                2,
            )

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if x <= left <= x + w and y <= top <= y + h:
                    cv2.putText(
                        frame,
                        name,
                        (x + 6, y + h - 6),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                    )
                    break

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
