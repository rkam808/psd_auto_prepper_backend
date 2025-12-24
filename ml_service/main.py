from fastapi import FastAPI, File, UploadFile, Query
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
from PIL import Image, ImageChops, ImageDraw

# --- 1. SETUP YUNET ---
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

if not os.path.exists(YUNET_FILE):
    try:
        urllib.request.urlretrieve(YUNET_URL, YUNET_FILE)
    except Exception as e:
        print(f"CRITICAL: Failed to download YuNet: {e}")

try:
    face_detector = cv2.FaceDetectorYN.create(
        YUNET_FILE, "", (320, 320), 0.6, 0.3, 5000
    )
    AI_AVAILABLE = True
except Exception:
    AI_AVAILABLE = False

# --- 2. MONKEY PATCH ---
try:
    import packbits
    codecs.packbits = packbits
except ImportError:
    pass

app = FastAPI()
session = new_session()

def get_smart_split_y(pil_img: Image.Image, padding_pct: float) -> int:
    width, height = pil_img.size
    split_y = int(height * 0.35)

    if AI_AVAILABLE:
        try:
            open_cv_image = np.array(pil_img)
            # YuNet requires BGR
            if open_cv_image.ndim == 3 and open_cv_image.shape[2] == 4:
                img_bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

            face_detector.setInputSize((width, height))
            _, faces = face_detector.detect(img_bgr)

            if faces is not None and len(faces) > 0:
                best_face = max(faces, key=lambda f: f[2] * f[3])
                _, y, _, h = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])

                # Default Padding: 0.02 (2% below chin)
                detected_y = y + h + int(h * padding_pct)
                split_y = max(10, min(detected_y, height - 10))

        except Exception as e:
            print(f"LOG: Detection Error: {e}")

    if split_y % 2 != 0:
        split_y += 1

    return int(split_y)

def generate_neck_extension(body_img: Image.Image, split_y: int, extension_ratio: float = 0.5) -> Image.Image:
    """
    Dynamically extends the neck upwards by a percentage of the Head Height (split_y).
    """
    width = body_img.width

    # 1. Calc Dynamic Height (50% of the head space)
    extension_height = int(split_y * extension_ratio)
    extension_height = max(50, min(extension_height, split_y))

    # 2. Grab Seed
    seed_row = body_img.crop((0, split_y, width, split_y + 1))

    # 3. Stretch
    neck_block = seed_row.resize((width, extension_height), resample=Image.Resampling.NEAREST)

    # 4. Fade Mask
    mask = Image.new('L', (width, extension_height), 0)
    draw = ImageDraw.Draw(mask)

    for y in range(extension_height):
        # 0 (Top) -> 255 (Bottom)
        opacity = int(255 * (y / extension_height))
        # Lock bottom 5% to solid
        if y > int(extension_height * 0.95):
            opacity = 255
        draw.line([(0, y), (width, y)], fill=opacity)

    r, g, b, a = neck_block.split()
    final_alpha = ImageChops.multiply(a, mask)
    neck_block = Image.merge('RGBA', (r, g, b, final_alpha))

    # 5. Paste
    extended_body = body_img.copy()
    paste_y = split_y - extension_height
    extended_body.paste(neck_block, (0, paste_y), mask=neck_block)

    return extended_body

def pack_layer(name: str, pil_image: Image.Image, x: int, y: int):
    w, h = pil_image.size
    if w == 0 or h == 0: return None

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
async def process_model(
    file: UploadFile = File(...),
    chin_padding: float = Query(0.02, description="Padding below chin (0.02 = 2%)"),
    inpaint_neck: bool = Query(True, description="Generate fake neck"),
    neck_ratio: float = Query(0.5, description="Height of ghost neck relative to head (0.5 = 50%)")
):
    print(f"\n--- PROCESSING (Padding: {chin_padding}, Neck Ratio: {neck_ratio}) ---")
    contents = await file.read()
    try:
        original_pil = Image.open(io.BytesIO(contents)).convert("RGBA")
    except:
        return Response(content="Invalid Image", status_code=400)

    # 1. Rembg
    character_only = remove(original_pil, session=session)
    width, height = character_only.size

    # 2. Smart Split
    split_y = get_smart_split_y(original_pil, chin_padding)
    print(f"LOG: Split Y = {split_y}")

    # 3. Create Layers
    body_full = character_only.copy()
    transparent_top = Image.new("RGBA", (width, split_y), (0, 0, 0, 0))
    body_full.paste(transparent_top, (0, 0))

    # DYNAMIC INPAINTING
    if inpaint_neck:
        body_full = generate_neck_extension(body_full, split_y, extension_ratio=neck_ratio)

    head_full = character_only.copy()
    transparent_bottom = Image.new("RGBA", (width, height - split_y), (0, 0, 0, 0))
    head_full.paste(transparent_bottom, (0, split_y))

    # 4. Pack
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
