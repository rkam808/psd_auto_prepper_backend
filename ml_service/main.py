from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import numpy as np
from rembg import remove, new_session
from pytoshop.user import nested_layers
from pytoshop import enums
from pytoshop import codecs
import io
import cv2
import os
import urllib.request
from PIL import Image

# --- 1. SETUP YUNET (Neural Network) ---
# This is a pre-trained face detector that runs on OpenCV's DNN module.
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

# Download model if missing
if not os.path.exists(YUNET_FILE):
    print("LOG: YuNet model not found. Downloading...")
    try:
        urllib.request.urlretrieve(YUNET_URL, YUNET_FILE)
        print("LOG: Download complete.")
    except Exception as e:
        print(f"CRITICAL: Failed to download YuNet: {e}")

# Initialize the Face Detector
try:
    # We initialize it with default params, we will update dimensions per image
    face_detector = cv2.FaceDetectorYN.create(
        YUNET_FILE,
        "",
        (320, 320), # Initial size, will change later
        0.6, # Score Threshold (Confidence)
        0.3, # NMS Threshold
        5000 # Top K
    )
    AI_AVAILABLE = True
    print("LOG: YuNet Face Detector loaded.")
except Exception as e:
    print(f"LOG: YuNet failed to load: {e}")
    AI_AVAILABLE = False


# --- 2. MONKEY PATCH FOR PSD ---
try:
    import packbits
    codecs.packbits = packbits
except ImportError:
    pass
# -------------------------------

app = FastAPI()
session = new_session()

def get_smart_split_y(pil_img: Image.Image) -> int:
    width, height = pil_img.size
    split_y = int(height * 0.35) # Default fallback

    if AI_AVAILABLE:
        try:
            # 1. Convert to Numpy/OpenCV format
            # YuNet expects BGR
            open_cv_image = np.array(pil_img)
            if open_cv_image.ndim == 3 and open_cv_image.shape[2] == 4:
                img_bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

            # 2. Update Detector Input Size
            # YuNet requires explicitly setting the input image size
            face_detector.setInputSize((width, height))

            # 3. Detect
            # faces is a numpy array: [x, y, w, h, ...]
            _, faces = face_detector.detect(img_bgr)

            # Check if faces were found
            if faces is not None and len(faces) > 0:
                # Pick the face with the highest confidence (Column 14 is score)
                # Face format: [x, y, w, h, x_re, y_re, x_le, y_le, ...]
                # We just want the largest one based on width*height
                best_face = max(faces, key=lambda f: f[2] * f[3])

                x, y, w, h = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
                score = best_face[14]

                print(f"LOG: Face Detected (Score: {score:.2f}) at: x={x}, y={y}, w={w}, h={h}")

                # --- TUNING ---
                # YuNet boxes are very accurate/tight to the chin.
                # We use 5% padding to safely clear the jawline.
                CHIN_ADJUSTMENT = 0.05

                detected_y = y + h + int(h * CHIN_ADJUSTMENT)

                split_y = max(10, min(detected_y, height - 10))
            else:
                print("LOG: No face detected by YuNet.")

        except Exception as e:
            print(f"LOG: Detection Error: {e}")

    # FORCE EVEN NUMBERS
    if split_y % 2 != 0:
        split_y += 1

    return int(split_y)

def pack_layer(name: str, pil_image: Image.Image, x: int, y: int):
    w, h = pil_image.size
    if w == 0 or h == 0: return None

    # FORCE EVEN COORDINATES
    if x % 2 != 0: x -= 1
    if y % 2 != 0: y -= 1

    channels = pil_image.split()
    if len(channels) == 4:
        r, g, b, a = channels
    else:
        r, g, b = channels
        a = Image.new('L', pil_image.size, 255)

    channels_dict = {
        0: np.array(r),
        1: np.array(g),
        2: np.array(b),
        -1: np.array(a)
    }

    layer = nested_layers.Image(
        name=name,
        visible=True,
        opacity=255,
        blend_mode=enums.BlendMode.normal,
        top=y, left=x, bottom=y + h, right=x + w,
        channels=channels_dict
    )
    return layer

@app.post("/process-model")
async def process_model(file: UploadFile = File(...)):
    print("\n--- PROCESSING START ---")
    contents = await file.read()
    try:
        original_pil = Image.open(io.BytesIO(contents)).convert("RGBA")
    except:
        return Response(content="Invalid Image", status_code=400)

    # 1. Rembg
    character_only = remove(original_pil, session=session)
    if character_only.getbbox() is None:
        return Response(content="Empty Image", status_code=422)

    width, height = character_only.size

    # 2. Smart Split (Using YuNet on Original Image)
    split_y = get_smart_split_y(original_pil)
    print(f"LOG: Split Y = {split_y}")

    # 3. Splits & PSD Gen
    body_full = character_only.copy()
    transparent_top = Image.new("RGBA", (width, split_y), (0, 0, 0, 0))
    body_full.paste(transparent_top, (0, 0))

    head_full = character_only.copy()
    transparent_bottom = Image.new("RGBA", (width, height - split_y), (0, 0, 0, 0))
    head_full.paste(transparent_bottom, (0, split_y))

    all_layers = []
    body_bbox = body_full.getbbox()
    if body_bbox:
        body_cropped = body_full.crop(body_bbox)
        all_layers.append(pack_layer("Body", body_cropped, x=body_bbox[0], y=body_bbox[1]))

    head_bbox = head_full.getbbox()
    if head_bbox:
        head_cropped = head_full.crop(head_bbox)
        all_layers.append(pack_layer("Head", head_cropped, x=head_bbox[0], y=head_bbox[1]))

    psd_file = nested_layers.nested_layers_to_psd(
        all_layers,
        color_mode=enums.ColorMode.rgb,
        depth=enums.ColorDepth.depth8,
        size=(height, width),
        compression=enums.Compression.rle
    )

    output = io.BytesIO()
    psd_file.write(output)
    output.seek(0)
    print("LOG: Done.\n")

    return Response(
        content=output.getvalue(),
        media_type="application/x-photoshop"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
