from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import numpy as np
from rembg import remove, new_session
from pytoshop.user import nested_layers
from pytoshop import enums
# 1. Import the codecs module to patch it
from pytoshop import codecs
import io
import sys
from PIL import Image

# --- CRITICAL FIX: MANUALLY INJECT PACKBITS ---
try:
    import packbits
    # Force pytoshop to use the external packbits library
    codecs.packbits = packbits
    print("DEBUG: 'packbits' successfully injected into pytoshop.")
except ImportError:
    print("CRITICAL ERROR: 'packbits' not found. Please run 'pip install packbits'")
# ----------------------------------------------

app = FastAPI()

session = new_session()

def pack_layer(name: str, pil_image: Image.Image, x: int, y: int):
    w, h = pil_image.size

    # Split channels
    channels = pil_image.split()
    if len(channels) == 4:
        r, g, b, a = channels
    else:
        r, g, b = channels
        a = Image.new('L', pil_image.size, 255)

    # Convert to standard numpy arrays (RLE handles the packing)
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
        top=y,
        left=x,
        bottom=y + h,
        right=x + w,
        channels=channels_dict
    )
    return layer

@app.post("/process-model")
async def process_model(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        original_pil = Image.open(io.BytesIO(contents)).convert("RGBA")
    except Exception:
        return Response(content="Invalid image data", status_code=400)

    # 1. Remove Background
    character_only = remove(original_pil, session=session)
    if character_only.getbbox() is None:
        return Response(content="Could not detect character in image", status_code=422)

    width, height = character_only.size

    # 2. CREATE SPLITS
    body_full = character_only.copy()
    transparent_block = Image.new("RGBA", (width, height // 2), (0, 0, 0, 0))
    body_full.paste(transparent_block, (0, 0))

    head_full = character_only.copy()
    transparent_block_bottom = Image.new("RGBA", (width, height - (height // 2)), (0, 0, 0, 0))
    head_full.paste(transparent_block_bottom, (0, height // 2))

    # 3. AUTO-CROP & PACK
    all_layers = []

    body_bbox = body_full.getbbox()
    if body_bbox:
        body_cropped = body_full.crop(body_bbox)
        all_layers.append(pack_layer("Body", body_cropped, x=body_bbox[0], y=body_bbox[1]))

    head_bbox = head_full.getbbox()
    if head_bbox:
        head_cropped = head_full.crop(head_bbox)
        all_layers.append(pack_layer("Head", head_cropped, x=head_bbox[0], y=head_bbox[1]))

    # 4. WRITE PSD
    psd_file = nested_layers.nested_layers_to_psd(
        all_layers,
        color_mode=enums.ColorMode.rgb,
        depth=enums.ColorDepth.depth8,
        # SIZE: (Height, Width) for correct Portrait orientation
        size=(height, width),
        # COMPRESSION: RLE
        # This is the industry standard. It fixes the "Cannot Open" (ZIP)
        # and "Shifted Pixels" (RAW) errors.
        compression=enums.Compression.rle
    )

    output = io.BytesIO()
    psd_file.write(output)
    output.seek(0)

    return Response(
        content=output.getvalue(),
        media_type="application/x-photoshop"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
