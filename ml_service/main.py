from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import numpy as np
import cv2
from rembg import remove
from pytoshop import layers
from pytoshop.enums import ColorMode
from pytoshop.core import PsdFile
import io
import os

app = FastAPI()

@app.post("/process-model")
async def process_model(file: UploadFile = File(...)):
    # Read image content
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    # Ensure image has an alpha channel for transparency
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Remove background
    img_no_bg = remove(img)

    # Image dimensions
    height, width, _ = img_no_bg.shape

    # Mock splitting into layers
    head_layer_img = img_no_bg[0:height//2, :]
    body_layer_img = img_no_bg[height//2:height, :]

    # Create PSD layers
    # Note: pytoshop expects RGB, but OpenCV uses BGR, so we convert.
    # rembg output is BGRA, so we convert to RGBA for pytoshop.
    head_layer = layers.ChannelImageData(image=cv2.cvtColor(head_layer_img, cv2.COLOR_BGRA2RGBA), is_visible=True)
    body_layer = layers.ChannelImageData(image=cv2.cvtColor(body_layer_img, cv2.COLOR_BGRA2RGBA), is_visible=True)

    layer_records = [
        layers.LayerRecord(
            name="Head",
            channels={-1: head_layer},
            top=0,
            left=0,
            bottom=head_layer_img.shape[0],
            right=head_layer_img.shape[1]
        ),
        layers.LayerRecord(
            name="Body",
            channels={-1: body_layer},
            top=head_layer_img.shape[0],
            left=0,
            bottom=height,
            right=width
        )
    ]

    # Construct PSD file
    psd_file = PsdFile(
        num_channels=4, height=height, width=width, color_mode=ColorMode.rgb, layer_records=layer_records
    )

    # Save PSD to a temporary file
    temp_psd_path = "temp.psd"
    with open(temp_psd_path, "wb") as f:
        psd_file.write(f)

    return FileResponse(
        temp_psd_path,
        media_type="image/vnd.adobe.photoshop",
        filename="prepped.psd",
    )

@app.on_event("shutdown")
def shutdown_event():
    if os.path.exists("temp.psd"):
        os.remove("temp.psd")