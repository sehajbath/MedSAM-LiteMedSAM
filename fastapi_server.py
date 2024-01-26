# filename: fastapi_server.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
import torch
from io import BytesIO

app = FastAPI()

# Assuming your model and its necessary functions are defined/imported here
from MedSAMLite_Model import MedSAM_Lite, load_model, preprocess_npz_image, postprocess_prediction

checkpoint_path = "work_dir/medsam_lite_latest.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@app.post("/process_npz/")
async def process_npz(file: UploadFile = File(...)):
    # Load the MedSAM_Lite model
    model = load_model(checkpoint_path, device)
    if file.filename.endswith('.npz'):
        contents = await file.read()
        npz_data = np.load(BytesIO(contents), allow_pickle=True)

        if 'imgs' in npz_data and 'gts' in npz_data:

            #output_image = postprocess_prediction(*preprocess_npz_image(model, npz_data, device))
            output_image = postprocess_prediction(*preprocess_npz_image(model, npz_data, device))
            # Return the image byte array directly
            return StreamingResponse(BytesIO(output_image), media_type="image/png")
        else:
            return {"error": "Invalid .npz file structure"}
    else:
        return {"error": "Invalid file format"}
