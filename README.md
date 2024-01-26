# README for MedSAM_Lite Application

## Overview

This README provides instructions and details for running the MedSAM_Lite application, which includes three main components: `streamlit_app.py`, `fastapi_server.py`, and `MedSAMLite_Model.py`. This application is designed to process medical images in NPZ format and predict segmentation masks using the MedSAM_Lite model.

## Setup and Installation

### Dependencies

To run the application, ensure the following dependencies are installed:

- Python 3.x
- Streamlit
- FastAPI
- Uvicorn
- PyTorch
- OpenCV (cv2)
- NumPy
- Matplotlib
- Pillow (PIL)
- BytesIO (from io)

### Installation Steps

1. Clone the repository or download the source code.
2. Install the required Python libraries:

   ```bash
   pip install streamlit fastapi uvicorn torch numpy opencv-python matplotlib pillow
   ```

3. Place the `tiny_vit_sam.py` and `segment_anything` module in the same directory as the main scripts, or ensure they are accessible in your Python path.

4. Ensure the model checkpoint file is available at the specified path in the scripts (e.g., `work_dir/medsam_lite_latest.pth`).

### Running the Application

- To test the inference efficiency:
   ```
  bash python inference_3D.py -data_root data/MedSAM_test/CT_Abd -pred_save_dir ./preds/CT_Abd -medsam_lite_checkpoint_path work_dir/medsam_lite_latest.pth -num_workers 4 --save_overlay -png_save_dir ./preds/CT_Abd_overlay --overwrite 
   ```

1. Start the FastAPI server:

   ```bash
   uvicorn fastapi_server:app --reload
   ```

2. In a separate terminal, run the Streamlit application:

   ```bash
   streamlit run streamlit_app.py
   ```

3. Access the Streamlit interface through a web browser at the address provided in the terminal (usually `http://localhost:8501`).

## Usage

1. On the Streamlit interface, upload an NPZ file containing medical images.
2. The file will be sent to the FastAPI server for processing.
3. The processed image with segmentation masks will be displayed on the Streamlit interface.

## Important Notes

- Ensure the model checkpoint file is compatible with the `MedSAM_Lite` model defined in `MedSAMLite_Model.py`.
- Modify file paths and configurations in the script as needed, based on your setup and file locations.
- The application is designed to run locally. For deployment or remote access, additional configuration might be necessary.
- The FastAPI server must be running for the Streamlit app to function correctly.

## Troubleshooting

- If you encounter dependency-related errors, verify that all required libraries are correctly installed and up-to-date.
- For issues related to model loading or processing, check the compatibility of the model checkpoint and ensure the correct model architecture is used.
- If the Streamlit app cannot connect to the FastAPI server, verify that the server is running and accessible on the specified port.

---

For further assistance or to report issues, please refer to the project's issue tracker or contact the maintainers.