from urllib import request
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from typing import Optional
from roop.core import (
    start,
    decode_execution_providers,
    suggest_max_memory,
    suggest_execution_threads,
)
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import normalize_output_path
import roop.globals
from PIL import Image
import tempfile
import os
import io
import random
import base64
from gender_recognition.model_utils import predict_gender


app = FastAPI(title="Face Swap API")

DEST_DIR = "dest"
# Point to your service account JSON
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/srivathsadhanraj/.gcp/face-swap-463306-863e4ddbf989.json"

# Initialize storage client
# try:
#     storage_client = storage.Client()
#     bucket = storage_client.bucket('face-swap-463306_cloudbuild')
# except Exception as e:
#     print(f"[ERROR] Failed to initialize GCS client: {e}")
#     bucket = None

def save_upload_to_temp(upload_file: UploadFile) -> str:
    image_bytes = upload_file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_file.name)
    return temp_file.name

# def fetch_random_from_gcs(gender: str, variant: str):
#     folder = f"{gender.lower()}/"
#     blobs = list(bucket.list_blobs(prefix=folder))
#     blobs = [b for b in blobs if b.name.lower().endswith((".jpg", ".jpeg", ".png"))]

#     if not blobs:
#         raise Exception(f"No images found in GCS folder '{folder}'")

#     # First try to filter by variant (if not 'surprise me')
#     filtered_blobs = blobs
#     if variant.lower() != "surprise me":
#         filtered_blobs = [b for b in blobs if variant.lower() in b.name.lower()]
#         if not filtered_blobs:
#             print(f"[WARN] No matches for variant '{variant}' in GCS. Falling back to random.")
#             filtered_blobs = blobs

#     chosen_blob = random.choice(filtered_blobs)

#     # Download as bytes
#     img_bytes = chosen_blob.download_as_bytes()
#     img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

#     # Save to temp file
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
#     img.save(temp_file.name)
#     return temp_file.name, os.path.basename(chosen_blob.name)


def choose_random_target_temp(variant: str, gender: str, source_type: str) -> str:
    if source_type == "remote":
        # return fetch_random_from_gcs(gender, variant)
        print("requested remote")

    gender_folder = os.path.join(DEST_DIR, gender.lower())
    if not os.path.exists(gender_folder):
        raise Exception(f"Destination gender folder '{gender_folder}' not found")

    files = [f for f in os.listdir(gender_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        raise Exception(f"No images found in '{gender_folder}'")

    filtered_files = files
    if variant.lower() != "surprise me":
        filtered_files = [f for f in files if variant.lower() in f.lower()]
        if not filtered_files:
            filtered_files = files

    chosen_file = random.choice(filtered_files)
    chosen_path = os.path.join(gender_folder, chosen_file)
    img = Image.open(chosen_path).convert("RGB")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(temp_file.name)

    return temp_file.name, chosen_file



def run_face_swap(source_path: str, target_path: str) -> str:
    roop.globals.source_path = source_path
    roop.globals.target_path = target_path
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    output_path = output_file.name
    roop.globals.output_path = normalize_output_path(source_path, target_path, output_path)

    roop.globals.frame_processors = ["face_swapper", "face_enhancer"]
    roop.globals.headless = True
    roop.globals.keep_fps = True
    roop.globals.keep_audio = True
    roop.globals.keep_frames = False
    roop.globals.many_faces = False
    roop.globals.video_encoder = "libx264"
    roop.globals.video_quality = 18
    roop.globals.max_memory = suggest_max_memory()
    roop.globals.execution_providers = decode_execution_providers(["cuda"])
    roop.globals.execution_threads = suggest_execution_threads()

    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_check():
            raise Exception("Pre-check failed for processor")

    val = start()
    return None if val == "No-Face" else output_path

@app.post("/swap-face")
async def swap_face_api(
    source_image: UploadFile = File(...),
    variant: str = Form("Surprise Me"),
    source_type: str = Form("local"),
    optional_target_image: Optional[UploadFile] = File(None)
):
    source_temp = target_temp = output_temp = None
    # fetch_random_from_gcs("male")
    # return {}
    try:
        source_temp = save_upload_to_temp(source_image)
        if optional_target_image:
            target_temp = save_upload_to_temp(optional_target_image)
            destination_name = optional_target_image.filename
        else:
            # Detect Gender from Source Image
            from PIL import Image as PILImage
            gender_result = predict_gender(PILImage.open(source_temp))
            gender = gender_result['gender'].lower()  # 'male' or 'female'

            print(f"[INFO] 1Detected Gender: {gender_result['gender']} ({gender_result['probability']*100:.2f}%)")

            target_temp, destination_name = choose_random_target_temp(variant, gender, source_type)

        output_temp = run_face_swap(source_temp, target_temp)
        if output_temp is None:
            raise Exception("No face detected in source image.")

        with open(output_temp, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Generate clean image ID from filename
        name_without_ext = os.path.splitext(destination_name)[0]
        image_id = name_without_ext.replace(" ", "_")

        return JSONResponse(status_code=200, content={
            "status": "success",
            "image_id": image_id,
            "image_base64": image_base64
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e)
        })

    finally:
        # Clean up all temp files
        for f in [source_temp, target_temp, output_temp]:
            if f and os.path.exists(f):
                os.remove(f)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc)},
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
