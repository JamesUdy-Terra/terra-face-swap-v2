from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import Response
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

app = FastAPI(title="Face Swap API")

DEST_DIR = "dest"

def save_upload_to_temp(upload_file: UploadFile) -> str:
    image_bytes = upload_file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_file.name)
    return temp_file.name

def choose_random_target_temp(variant: str) -> str:
    if not os.path.exists(DEST_DIR):
        raise Exception(f"Destination folder '{DEST_DIR}' not found")

    # Get all supported image files
    files = [
        f for f in os.listdir(DEST_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not files:
        raise Exception(f"No destination images found in '{DEST_DIR}'")

    filtered_files = files

    # Filter if not "Surprise Me"
    if variant.lower() != "surprise me":
        filtered_files = [
            f for f in files
            if variant.lower() in f.lower()
        ]

        # Fallback to random from all files if nothing matched
        if not filtered_files:
            print(f"[WARN] No matches for variant '{variant}'. Falling back to random.")
            filtered_files = files

    # Choose a random image from filtered (or fallback) list
    chosen_file = random.choice(filtered_files)
    chosen_path = os.path.join(DEST_DIR, chosen_file)

    # Open and normalize it (RGB + .jpg) into a temp file
    img = Image.open(chosen_path).convert("RGB")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(temp_file.name)

    return temp_file.name

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

    start()
    return output_path

@app.post("/swap-face")
async def swap_face_api(source_image: UploadFile = File(...), variant: str = Form("Surprise Me")):
    source_temp = target_temp = output_temp = None
    try:
        source_temp = save_upload_to_temp(source_image)
        target_temp = choose_random_target_temp(variant)
        output_temp = run_face_swap(source_temp, target_temp)

        with open(output_temp, "rb") as f:
            image_bytes = f.read()

        return Response(content=image_bytes, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up all temp files
        for f in [source_temp, target_temp, output_temp]:
            if f and os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
