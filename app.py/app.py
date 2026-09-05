import os
import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace

st.set_page_config(page_title="Face Recognition App", layout="centered")

st.title("Face Recognition Web Application")
st.write("Take a photo using your webcam to verify your identity.")

KNOWN_FACES_DIR = "known_faces"

# Webcam camera input
picture = st.camera_input("Take a snapshot")

if picture:
    # 1. Convert snapshot to an OpenCV image array
    bytes_data = picture.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # 2. Save image temporarily for DeepFace analysis
    temp_path = "temp_capture.jpg"
    cv2.imwrite(temp_path, img)

    st.info("Analyzing face features...")

    best_match = "Unknown"
    match_found = False

    # 3. Compare with images in the known_faces directory
    if os.path.exists(KNOWN_FACES_DIR):
        for file in os.listdir(KNOWN_FACES_DIR):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                ref_path = os.path.join(KNOWN_FACES_DIR, file)

                try:
                    # Compare facial embeddings using DeepFace
                    result = DeepFace.verify(
                        img1_path=temp_path,
                        img2_path=ref_path,
                        model_name="VGG-Face",
                        enforce_detection=True,
                    )

                    if result.get("verified", False):
                        # Extract person's name from filename (e.g., YourName.jpg -> YourName)
                        best_match = os.path.splitext(file)[0]
                        match_found = True
                        break
                except Exception:
                    # Skip if face detection fails in either reference or captured photo
                    continue

    # 4. Remove temporary capture file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # 5. Output result
    if match_found:
        st.success(f"Person Identified: **{best_match}**")
    else:
        st.warning("Person Identified: **Unknown**")