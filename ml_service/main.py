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
from PIL import Image, ImageChops, ImageDraw, ImageFilter

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
    pts = np.array(points)
    if pts.shape[1] > 2: pts = pts[:, :2]

    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)

    w = max_x - min_x
    h = max_y - min_y

    if w < min_size: w = min_size
    if h < min_size: h = min_size

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

def get_skin_tone_and_shadow(image: Image.Image, mouth_pts) -> tuple:
    try:
        pts = np.array(mouth_pts)
        if pts.shape[1] > 2: pts = pts[:, :2]

        mouth_center_x = int(np.mean(pts[:, 0]))
        mouth_bottom_y = int(np.max(pts[:, 1]))

        sample_x = mouth_center_x
        sample_y = mouth_bottom_y + 20
        r = 8
        box = (sample_x - r, sample_y - r, sample_x + r, sample_y + r)

        if box[0] < 0 or box[1] < 0 or box[2] > image.width or box[3] > image.height:
             box = (sample_x - r, mouth_bottom_y, sample_x + r, mouth_bottom_y + 10)

        sample = image.crop(box)
        arr = np.array(sample)

        fallback = (255, 224, 189, 255)
        fallback_shadow = (215, 184, 159, 255)

        if arr.size == 0: return fallback, fallback_shadow
        valid_pixels = arr[arr[:, :, 3] > 0]
        if len(valid_pixels) == 0: return fallback, fallback_shadow

        brightness = np.mean(valid_pixels[:, :3], axis=1)
        sorted_indices = np.argsort(brightness)

        base_indices = sorted_indices[int(len(sorted_indices) * 0.5) : int(len(sorted_indices) * 0.9)]
        if len(base_indices) > 0:
            base_color = np.median(valid_pixels[base_indices], axis=0).astype(int)
        else:
            base_color = np.median(valid_pixels, axis=0).astype(int)

        shadow_start = int(len(sorted_indices) * 0.10)
        shadow_end = int(len(sorted_indices) * 0.40)
        shadow_indices = sorted_indices[shadow_start:shadow_end]

        if len(shadow_indices) > 0:
            shadow_candidate = np.median(valid_pixels[shadow_indices], axis=0).astype(int)
        else:
            shadow_candidate = base_color

        base_v = np.mean(base_color[:3])
        shadow_v = np.mean(shadow_candidate[:3])

        if (base_v - shadow_v) < 15:
            syn_shadow = (base_color[:3] * 0.85).astype(int)
            shadow_color = (syn_shadow[0], syn_shadow[1], syn_shadow[2], 255)
            print(f"LOG: Synthetic Shadow Generated (Flat Art Detected)")
        elif shadow_v < 50:
            syn_shadow = (base_color[:3] * 0.85).astype(int)
            shadow_color = (syn_shadow[0], syn_shadow[1], syn_shadow[2], 255)
            print(f"LOG: Synthetic Shadow Generated (Sample was too dark/black)")
        else:
            shadow_color = shadow_candidate
            print(f"LOG: Sampled Shadow Found (Diff: {base_v - shadow_v})")

        return tuple(base_color), tuple(shadow_color)

    except Exception as e:
        print(f"Color Sample Error: {e}")
        return (255, 224, 189, 255), (215, 184, 159, 255)

def heal_hole(image: Image.Image, box: tuple, fill_color: tuple) -> Image.Image:
    x1, y1, x2, y2 = box
    healed_img = image.copy()
    draw = ImageDraw.Draw(healed_img)
    draw.rectangle((x1 - 1, y1 - 1, x2 + 1, y2 + 1), fill=fill_color)
    return healed_img

def get_feature_boxes_spatial(face: FaceData):
    all_pts = np.array(face.landmarks)
    if all_pts.shape[1] > 2: all_pts = all_pts[:, :2]

    sorted_by_y = all_pts[all_pts[:, 1].argsort()]

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

    eyes_and_brows = sorted_by_y[:-8]

    sorted_by_x = eyes_and_brows[eyes_and_brows[:, 0].argsort()]
    mid_idx = len(sorted_by_x) // 2
    right_cluster = sorted_by_x[:mid_idx]
    left_cluster = sorted_by_x[mid_idx:]

    def get_eye_subset(cluster):
        c_sorted = cluster[cluster[:, 1].argsort()]
        return c_sorted[-6:]

    pts_eye_r = get_eye_subset(right_cluster)
    pts_eye_l = get_eye_subset(left_cluster)

    return pts_eye_r, pts_eye_l, pts_mouth

def extract_bangs_by_color(image: Image.Image, skin_color: tuple, exclusion_boxes: list, eye_r_box=None, eye_l_box=None):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    lower_skin_1 = np.array([0, 15, 60], dtype=np.uint8)
    upper_skin_1 = np.array([25, 255, 255], dtype=np.uint8)
    lower_skin_2 = np.array([160, 15, 60], dtype=np.uint8)
    upper_skin_2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_skin_1, upper_skin_1)
    mask2 = cv2.inRange(hsv, lower_skin_2, upper_skin_2)
    skin_mask = mask1 | mask2

    alpha = np.array(image)[:, :, 3]
    skin_mask[alpha == 0] = 0

    kernel = np.ones((4,4), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    hair_mask = (alpha > 0) & (skin_mask == 0)

    for box in exclusion_boxes:
        x1, y1, x2, y2 = box
        hair_mask[y1:y2, x1:x2] = False

    img_arr = np.array(image)

    hair_layer_arr = np.zeros_like(img_arr)
    hair_layer_arr[hair_mask] = img_arr[hair_mask]
    hair_layer = Image.fromarray(hair_layer_arr)

    kernel_erode = np.ones((3,3), np.uint8)
    eroded_skin_mask = cv2.erode(skin_mask, kernel_erode, iterations=2)

    cut_mask = (alpha > 0) & (eroded_skin_mask == 0)

    for box in exclusion_boxes:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        cut_mask[y1:y2, x1:x2] = False

    face_only_arr = img_arr.copy()
    face_only_arr[cut_mask] = 0
    face_only = Image.fromarray(face_only_arr)

    cranium_img = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(cranium_img)

    if eye_r_box and eye_l_box:
        left_edge = eye_r_box[0]
        right_edge = eye_l_box[2]
        top_edge = min(eye_r_box[1], eye_l_box[1])

        face_width = right_edge - left_edge
        center_x = left_edge + (face_width / 2)

        cranium_w = face_width * 1.3
        cranium_h = face_width * 1.2

        e_x1 = center_x - (cranium_w / 2)
        e_x2 = center_x + (cranium_w / 2)
        e_y1 = top_edge - (cranium_h * 0.8)
        e_y2 = top_edge + (cranium_h * 0.5)

        draw.ellipse((e_x1, e_y1, e_x2, e_y2), fill=skin_color)

    healed_base = Image.alpha_composite(cranium_img, face_only)

    return healed_base, hair_layer


def extract_features(head_layer: Image.Image, face: FaceData):

    pts_eye_r, pts_eye_l, pts_mouth = get_feature_boxes_spatial(face)

    skin_color, shadow_color = get_skin_tone_and_shadow(head_layer, pts_mouth)
    print(f"LOG: Base Skin: {skin_color}, Shadow Skin: {shadow_color}")

    re_box = get_directional_box(pts_eye_r, pad_x=0.1, pad_top=0.6, pad_bottom=0.3)
    le_box = get_directional_box(pts_eye_l, pad_x=0.1, pad_top=0.6, pad_bottom=0.3)
    m_box = get_directional_box(pts_mouth, pad_x=0.4, pad_top=0.3, pad_bottom=0.3)

    eye_r_layer = head_layer.crop(re_box)
    eye_l_layer = head_layer.crop(le_box)
    mouth_layer = head_layer.crop(m_box)

    head_skin_base = head_layer.copy()
    head_skin_base = heal_hole(head_skin_base, re_box, skin_color)
    head_skin_base = heal_hole(head_skin_base, le_box, skin_color)
    head_skin_base = heal_hole(head_skin_base, m_box, skin_color)

    exclusion_list = [m_box]

    bald_head, hair_layer = extract_bangs_by_color(
        head_skin_base,
        skin_color,
        exclusion_list,
        eye_r_box=re_box,
        eye_l_box=le_box
    )

    face_width = le_box[2] - re_box[0]
    return bald_head, (eye_l_layer, le_box), (eye_r_layer, re_box), (mouth_layer, m_box), hair_layer, skin_color, shadow_color, face_width

def scan_neck_geometry(body_img: Image.Image, split_y: int, skin_color: tuple, shadow_color: tuple, threshold=40):
    """
    Scans the top of the body layer to find where the skin/shadow pixels start and end.
    Returns (neck_center, neck_width).
    """
    # Crop a small strip (10px height) just below the cut line
    # We go down +5px to avoid anti-aliasing artifacts at the exact cut line
    scan_y = split_y + 5
    scan_h = 10

    if scan_y + scan_h > body_img.height:
        # Image too short? fallback.
        return None, None

    strip = body_img.crop((0, scan_y, body_img.width, scan_y + scan_h))
    arr = np.array(strip)

    # Extract channels
    r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]

    # Helper to calculate mask for a color target
    def get_mask(target):
        tr, tg, tb = target[:3]
        # Euclidean distance
        dist = np.sqrt(
            (r.astype(int) - tr)**2 +
            (g.astype(int) - tg)**2 +
            (b.astype(int) - tb)**2
        )
        return (dist < threshold) & (a > 0)

    # Create masks for Base Skin AND Shadow Skin
    mask_skin = get_mask(skin_color)
    mask_shadow = get_mask(shadow_color)

    # Combine: valid pixel is either skin OR shadow
    final_mask = mask_skin | mask_shadow

    # Find all X coordinates that are True
    # We sum vertically (axis 0) to check if ANY pixel in the column matched
    col_hits = np.any(final_mask, axis=0)

    # Get indices of True columns
    valid_indices = np.where(col_hits)[0]

    if len(valid_indices) == 0:
        return None, None # No skin found (Turtleneck?)

    # Get Bounds
    x_min = valid_indices[0]
    x_max = valid_indices[-1]

    measured_width = x_max - x_min
    center = x_min + (measured_width // 2)

    print(f"LOG: Neck Scan Successful. Range: {x_min}-{x_max} (Width: {measured_width})")
    return center, measured_width

def generate_neck_extension(
    body_img: Image.Image,
    split_y: int,
    neck_color: tuple,
    skin_color_ref: tuple, # Needed for scanning
    face_width: int,
    width_ratio: float = 0.65,
    extension_ratio: float = 0.5
) -> Image.Image:

    width = body_img.width
    height = body_img.height

    # 1. Try to SCAN for exact neck position
    scan_center, scan_width = scan_neck_geometry(body_img, split_y, skin_color_ref, neck_color)

    if scan_center is not None and scan_width > 20:
        # Scan successful! Use measured data.
        neck_w = int(scan_width * 0.95) # Shrink slightly to fit INSIDE the lines
        center_x = scan_center
    else:
        # Fallback to ratio math (Turtleneck or failed scan)
        print("LOG: Neck Scan Failed (No skin found). Using Ratio Fallback.")
        neck_w = int(face_width * width_ratio)
        center_x = width // 2

    neck_x1 = center_x - (neck_w // 2)
    neck_x2 = center_x + (neck_w // 2)

    extension_height = int(split_y * extension_ratio)
    extension_height = max(50, min(extension_height, split_y))

    neck_layer = Image.new("RGBA", (width, extension_height + 20), (0, 0, 0, 0))
    draw = ImageDraw.Draw(neck_layer)

    draw.rectangle((neck_x1, 0, neck_x2, extension_height + 20), fill=neck_color)

    body_content = body_img.crop((0, split_y, width, height))
    final_body = Image.new("RGBA", (width, height), (0,0,0,0))

    neck_paste_y = split_y - extension_height
    final_body.paste(neck_layer, (0, neck_paste_y))
    final_body.paste(body_content, (0, split_y), mask=body_content)

    return final_body

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
    neck_width_scale: float = Query(0.65),
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

    # 3. Head Prep
    head_full = character_only.copy()
    transparent_bottom = Image.new("RGBA", (width, height - split_y), (0, 0, 0, 0))
    head_full.paste(transparent_bottom, (0, split_y))

    # Defaults
    global_skin_color = (255, 224, 189, 255)
    global_shadow_color = (215, 184, 159, 255)
    global_face_width = 300

    all_layers = []

    if separate_features and face_data:
        bald_head, eye_l, eye_r, mouth, hair_layer, skin_c, shadow_c, face_w = extract_features(head_full, face_data)
        global_skin_color = skin_c
        global_shadow_color = shadow_c
        global_face_width = face_w

        head_bbox = bald_head.getbbox()
        if head_bbox:
            all_layers.append(pack_layer("Face_Skin", bald_head.crop(head_bbox), head_bbox[0], head_bbox[1]))
        all_layers.append(pack_layer("Eye_L", eye_l[0], eye_l[1][0], eye_l[1][1]))
        all_layers.append(pack_layer("Eye_R", eye_r[0], eye_r[1][0], eye_r[1][1]))
        all_layers.append(pack_layer("Mouth", mouth[0], mouth[1][0], mouth[1][1]))
        hair_bbox = hair_layer.getbbox()
        if hair_bbox:
            all_layers.append(pack_layer("Hair_Front", hair_layer.crop(hair_bbox), hair_bbox[0], hair_bbox[1]))
    else:
        head_bbox = head_full.getbbox()
        if head_bbox:
            all_layers.append(pack_layer("Head", head_full.crop(head_bbox), head_bbox[0], head_bbox[1]))

    # 4. Body with SCANNED NECK
    body_full = character_only.copy()
    transparent_top = Image.new("RGBA", (width, split_y), (0, 0, 0, 0))
    body_full.paste(transparent_top, (0, 0))

    if inpaint_neck:
        body_full = generate_neck_extension(
            body_full,
            split_y,
            neck_color=global_shadow_color,
            skin_color_ref=global_skin_color, # Pass Scan Ref
            face_width=global_face_width,
            width_ratio=neck_width_scale,
            extension_ratio=neck_ratio
        )

    body_bbox = body_full.getbbox()
    if body_bbox:
        all_layers.insert(0, pack_layer("Body", body_full.crop(body_bbox), body_bbox[0], body_bbox[1]))

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
