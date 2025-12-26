from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import Response
import numpy as np
from rembg import remove, new_session
from pytoshop.user import nested_layers
from pytoshop import enums
from pytoshop import codecs
import io
import os
import cv2
from PIL import Image, ImageChops, ImageDraw

# --- 1. SETUP ADVANCED DETECTOR ---
try:
    import torch
    from anime_face_detector import create_detector
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"LOG: Loading Advanced Detector on {device}...")
    detector = create_detector('yolov3', device=device)
    ADVANCED_AI = True
    print("LOG: Advanced Anime Detector Loaded (28 Landmarks).")
except Exception as e:
    print(f"LOG: Advanced libraries missing or failed: {e}")
    ADVANCED_AI = False

# --- 2. MONKEY PATCH ---
try:
    import packbits
    codecs.packbits = packbits
except ImportError:
    pass

app = FastAPI()
session = new_session()

# --- HELPER CLASSES ---
class FaceData:
    def __init__(self, x, y, w, h, landmarks):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.landmarks = landmarks

def get_directional_box(points, pad_x=0.2, pad_top=0.5, pad_bottom=0.2, min_size=10):
    """
    Creates a box with uneven padding.
    """
    pts = np.array(points)
    if pts.shape[1] > 2: pts = pts[:, :2]

    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)

    w = max_x - min_x
    h = max_y - min_y

    # Safety minimum
    if w < min_size: w = min_size
    if h < min_size: h = min_size

    # Apply Directional Padding
    # pad_x is TOTAL width padding (split left/right)
    # e.g. 0.1 = 5% left, 5% right

    extra_w = w * pad_x
    extra_top = h * pad_top
    extra_btm = h * pad_bottom

    x1 = int(min_x - (extra_w / 2))
    x2 = int(max_x + (extra_w / 2))

    y1 = int(min_y - extra_top)
    y2 = int(max_y + extra_btm)

    return (x1, y1, x2, y2)

def analyze_face(pil_img: Image.Image) -> FaceData | None:
    if not ADVANCED_AI: return None
    try:
        open_cv_image = np.array(pil_img)
        img_bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGR)
        preds = detector(img_bgr)
        if len(preds) > 0:
            face = max(preds, key=lambda p: p['bbox'][4])
            x1, y1, x2, y2, _ = face['bbox']
            return FaceData(x1, y1, x2-x1, y2-y1, face['keypoints'])
    except Exception as e:
        print(f"LOG: Analysis Error: {e}")
    return None

def get_skin_tone(image: Image.Image, mouth_pts) -> tuple:
    try:
        pts = np.array(mouth_pts)
        if pts.shape[1] > 2: pts = pts[:, :2]

        mouth_center_x = int(np.mean(pts[:, 0]))
        mouth_bottom_y = int(np.max(pts[:, 1]))

        # Move down 20px (Chin area)
        sample_x = mouth_center_x
        sample_y = mouth_bottom_y + 20

        r = 6
        box = (sample_x - r, sample_y - r, sample_x + r, sample_y + r)

        if box[0] < 0 or box[1] < 0 or box[2] > image.width or box[3] > image.height:
             box = (sample_x - r, mouth_bottom_y, sample_x + r, mouth_bottom_y + 10)

        sample = image.crop(box)
        arr = np.array(sample)

        if arr.size == 0: return (255, 224, 189, 255)
        valid_pixels = arr[arr[:, :, 3] > 0]
        if len(valid_pixels) == 0: return (255, 224, 189, 255)

        brightness = np.mean(valid_pixels[:, :3], axis=1)
        sorted_indices = np.argsort(brightness)
        bright_indices = sorted_indices[int(len(sorted_indices) * 0.5):]

        if len(bright_indices) > 0:
            skin_color = np.median(valid_pixels[bright_indices], axis=0).astype(int)
        else:
            skin_color = np.median(valid_pixels, axis=0).astype(int)

        return tuple(skin_color)
    except Exception:
        return (255, 224, 189, 255)

def heal_hole(image: Image.Image, box: tuple, fill_color: tuple) -> Image.Image:
    x1, y1, x2, y2 = box
    healed_img = image.copy()
    draw = ImageDraw.Draw(healed_img)
    draw.rectangle((x1 - 1, y1 - 1, x2 + 1, y2 + 1), fill=fill_color)
    return healed_img

def get_feature_boxes_spatial(face: FaceData):
    all_pts = np.array(face.landmarks)
    if all_pts.shape[1] > 2: all_pts = all_pts[:, :2]

    # 1. Sort by Y (Top to Bottom)
    sorted_by_y = all_pts[all_pts[:, 1].argsort()]

    # 2. CLEAN THE MOUTH (Remove Chin)
    bottom_5 = sorted_by_y[-5:]
    bottom_5 = bottom_5[bottom_5[:, 1].argsort()]
    lowest = bottom_5[-1]
    second_lowest = bottom_5[-2]
    gap = abs(lowest[1] - second_lowest[1])

    face_h_est = (sorted_by_y[-1][1] - sorted_by_y[0][1])
    if gap > (face_h_est * 0.05):
        pts_mouth = bottom_5[:-1]
    else:
        pts_mouth = bottom_5

    # 3. Get Eyes (Excluding Nose/Mouth/Chin)
    eyes_and_brows = sorted_by_y[:-8]

    # 4. Split Left vs Right
    sorted_by_x = eyes_and_brows[eyes_and_brows[:, 0].argsort()]
    mid_idx = len(sorted_by_x) // 2
    right_cluster = sorted_by_x[:mid_idx] # Viewer Left
    left_cluster = sorted_by_x[mid_idx:]  # Viewer Right

    # 5. Extract Eyes from Brows (Bottom 6 of each cluster)
    def get_eye_subset(cluster):
        c_sorted = cluster[cluster[:, 1].argsort()]
        return c_sorted[-6:]

    pts_eye_r = get_eye_subset(right_cluster)
    pts_eye_l = get_eye_subset(left_cluster)

    return pts_eye_r, pts_eye_l, pts_mouth

def extract_features(head_layer: Image.Image, face: FaceData):

    pts_eye_r, pts_eye_l, pts_mouth = get_feature_boxes_spatial(face)

    skin_color = get_skin_tone(head_layer, pts_mouth)

    # --- UPDATED PADDING ---
    # Eyes: pad_x REDUCED from 0.3 to 0.1 (Tighter Sides)
    # Top/Bottom kept same (0.6 / 0.3) as they work well
    re_box = get_directional_box(pts_eye_r, pad_x=0.1, pad_top=0.6, pad_bottom=0.3)
    le_box = get_directional_box(pts_eye_l, pad_x=0.1, pad_top=0.6, pad_bottom=0.3)

    # Mouth: Tight Top/Bottom (0.3), Wide Sides (0.4)
    m_box = get_directional_box(pts_mouth, pad_x=0.4, pad_top=0.3, pad_bottom=0.3)

    print(f"LOG: Eye R Box: {re_box}")
    print(f"LOG: Eye L Box: {le_box}")

    eye_r_layer = head_layer.crop(re_box)
    eye_l_layer = head_layer.crop(le_box)
    mouth_layer = head_layer.crop(m_box)

    healed_head = head_layer.copy()
    healed_head = heal_hole(healed_head, re_box, skin_color)
    healed_head = heal_hole(healed_head, le_box, skin_color)
    healed_head = heal_hole(healed_head, m_box, skin_color)

    return healed_head, (eye_l_layer, le_box), (eye_r_layer, re_box), (mouth_layer, m_box)

# --- REUSED HELPERS ---
def generate_neck_extension(body_img: Image.Image, split_y: int, extension_ratio: float = 0.5) -> Image.Image:
    width = body_img.width
    extension_height = int(split_y * extension_ratio)
    extension_height = max(50, min(extension_height, split_y))
    seed_row = body_img.crop((0, split_y, width, split_y + 1))
    neck_block = seed_row.resize((width, extension_height), resample=Image.Resampling.NEAREST)
    mask = Image.new('L', (width, extension_height), 0)
    draw = ImageDraw.Draw(mask)
    for y in range(extension_height):
        opacity = int(255 * (y / extension_height))
        if y > int(extension_height * 0.95): opacity = 255
        draw.line([(0, y), (width, y)], fill=opacity)
    r, g, b, a = neck_block.split()
    final_alpha = ImageChops.multiply(a, mask)
    neck_block = Image.merge('RGBA', (r, g, b, final_alpha))
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
    if len(channels) == 4: r, g, b, a = channels
    else: r, g, b = channels; a = Image.new('L', pil_image.size, 255)
    channels_dict = {0: np.array(r), 1: np.array(g), 2: np.array(b), -1: np.array(a)}
    return nested_layers.Image(name=name, visible=True, opacity=255, blend_mode=enums.BlendMode.normal, top=y, left=x, bottom=y + h, right=x + w, channels=channels_dict)

@app.post("/process-model")
async def process_model(
    file: UploadFile = File(...),
    chin_padding: float = Query(0.02),
    inpaint_neck: bool = Query(True),
    neck_ratio: float = Query(0.5),
    separate_features: bool = Query(True)
):
    print(f"\n--- PROCESSING (Advanced AI: {ADVANCED_AI}) ---")
    contents = await file.read()
    try:
        original_pil = Image.open(io.BytesIO(contents)).convert("RGBA")
    except:
        return Response(content="Invalid Image", status_code=400)

    # 1. Rembg
    character_only = remove(original_pil, session=session)
    width, height = character_only.size

    # 2. Analyze
    face_data = analyze_face(original_pil)

    split_y = int(height * 0.35)
    if face_data:
        detected_y = face_data.y + face_data.h + int(face_data.h * chin_padding)
        split_y = max(10, min(detected_y, height - 10))
    if split_y % 2 != 0: split_y += 1
    print(f"LOG: Split Y = {split_y}")

    # 3. Body
    body_full = character_only.copy()
    transparent_top = Image.new("RGBA", (width, split_y), (0, 0, 0, 0))
    body_full.paste(transparent_top, (0, 0))
    if inpaint_neck:
        body_full = generate_neck_extension(body_full, split_y, extension_ratio=neck_ratio)

    # 4. Head
    head_full = character_only.copy()
    transparent_bottom = Image.new("RGBA", (width, height - split_y), (0, 0, 0, 0))
    head_full.paste(transparent_bottom, (0, split_y))

    all_layers = []
    body_bbox = body_full.getbbox()
    if body_bbox:
        all_layers.append(pack_layer("Body", body_full.crop(body_bbox), body_bbox[0], body_bbox[1]))

    # 5. Features
    if separate_features and face_data:
        healed_head, eye_l, eye_r, mouth = extract_features(head_full, face_data)

        head_bbox = healed_head.getbbox()
        if head_bbox:
            all_layers.append(pack_layer("Head_Base", healed_head.crop(head_bbox), head_bbox[0], head_bbox[1]))

        all_layers.append(pack_layer("Eye_L", eye_l[0], eye_l[1][0], eye_l[1][1]))
        all_layers.append(pack_layer("Eye_R", eye_r[0], eye_r[1][0], eye_r[1][1]))
        all_layers.append(pack_layer("Mouth", mouth[0], mouth[1][0], mouth[1][1]))
    else:
        head_bbox = head_full.getbbox()
        if head_bbox:
            all_layers.append(pack_layer("Head", head_full.crop(head_bbox), head_bbox[0], head_bbox[1]))

    # 6. Save
    psd_file = nested_layers.nested_layers_to_psd(
        all_layers, color_mode=enums.ColorMode.rgb, depth=enums.ColorDepth.depth8,
        size=(height, width), compression=enums.Compression.rle
    )
    output = io.BytesIO()
    psd_file.write(output)
    output.seek(0)
    print("LOG: Done.\n")

    return Response(content=output.getvalue(), media_type="application/x-photoshop")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
