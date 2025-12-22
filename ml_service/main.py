from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import numpy as np
import cv2
from rembg import remove, new_session
from pytoshop.user import nested_layers
from pytoshop import enums
import io

app = FastAPI()

# Initialize the model once
session = new_session("isnet-anime")

def pack_layer(name: str, img_array: np.ndarray):
    """
    Helper: Converts a numpy RGBA image into a Pytoshop nested_layers.Image.
    """
    height, width, channels = img_array.shape

    if channels == 4:
        b, g, r, a = cv2.split(img_array)
    elif channels == 3:
        b, g, r = cv2.split(img_array)
        a = np.ones((height, width), dtype=np.uint8) * 255
    else: # Grayscale
        b = g = r = img_array
        a = np.ones((height, width), dtype=np.uint8) * 255


    channels_dict = {0: r, 1: g, 2: b, -1: a}

    layer = nested_layers.Image(
        name=name,
        visible=True,
        opacity=255,
        blend_mode=enums.BlendMode.normal,
        top=0,
        left=0,
        channels=channels_dict
    )
    return layer

@app.post("/process-model")
async def process_model(file: UploadFile = File(...)):
    # 1. Read the uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # Ensure we have 4 channels (BGRA) for consistency
    if len(original_img.shape) == 2: # Grayscale
         original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGRA)
    elif original_img.shape[2] == 3: # RGB/BGR
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2BGRA)

    # 2. Remove Background using Rembg
    character_only = remove(original_img, session=session)
    height, width, _ = character_only.shape

    # 3. SPLIT LOGIC (Mockup: Top Half vs Bottom Half)
    head_img = character_only.copy()
    head_img[height//2:, :] = 0  # Mask bottom half

    body_img = character_only.copy()
    body_img[:height//2, :] = 0  # Mask top half

    # 4. Construct the PSD
    all_layers = [
        pack_layer("Body", body_img), # bottom layer
        pack_layer("Head", head_img)  # top layer
    ]

    # nested_layers_to_psd will calculate the size, but it's good to be explicit
    psd_file = nested_layers.nested_layers_to_psd(
        all_layers,
        color_mode=enums.ColorMode.rgb,
        depth=enums.ColorDepth.depth8,
        size=(height, width),
        compression=enums.Compression.zip
    )

    # 5. Write to Bytes
    output = io.BytesIO()
    psd_file.write(output)
    output.seek(0)

    # We use Response (instead of FileResponse) because the file is in RAM, not on disk
    return Response(
        content=output.getvalue(),
        media_type="application/x-photoshop"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)