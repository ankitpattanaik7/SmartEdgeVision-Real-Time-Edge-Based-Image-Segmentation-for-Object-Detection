# Advanced Feature Extraction and Image Segmentation using Canny Edge Detection with Interactive Streamlit Web Interface

## What This Project Does
This project is an end-to-end computer vision application that:
- extracts image features using Canny edge detection,
- segments objects/regions using contour-based and region-based methods,
- compares a custom academic Canny implementation with OpenCV's built-in Canny,
- provides an interactive Streamlit interface for experimentation and analysis.

The app is designed so a new user can upload an image, tune parameters, view intermediate processing stages, compare segmentation outputs, and download results.

## Real Problem It Solves
In many communities and campuses, pavement damage (cracks, broken edges, potholes) is common, but inspection is manual and inconsistent. Teams often receive photos from phones under varying light and quality conditions. Boundary detection and damage area estimation become subjective and slow.

This project helps by providing:
- consistent edge extraction,
- visual segmentation comparisons,
- quick parameter tuning for different imaging conditions,
- downloadable outputs for reports and maintenance planning.

## Why It Is Important
- Improves safety by helping locate high-risk damaged zones.
- Supports preventive maintenance before severe deterioration.
- Reduces manual effort in image-based inspections.
- Provides transparent algorithmic understanding for academic work.
- Produces reproducible outputs for technical reporting.

## Core Features
- Full custom Canny pipeline:
  - Gaussian smoothing
  - Sobel gradient computation
  - Non-maximum suppression
  - Double thresholding
  - Hysteresis edge tracking
- OpenCV Canny baseline for performance comparison
- Segmentation:
  - contour-based segmentation
  - region-growing segmentation
  - watershed segmentation
- Streamlit interface:
  - image upload
  - live parameter controls
  - intermediate stage visualization
  - processed output downloads
  - runtime and sensitivity analysis
  - webcam edge overlay (if webcam dependencies are available)

## Project Structure
```text
.
|-- main.py            # Streamlit app and UI workflow
|-- canny.py           # Custom Canny implementation with intermediate outputs
|-- segmentation.py    # Contour, region-growing, and watershed segmentation
|-- utils.py           # Image helpers, plotting helpers, encoding utilities
|-- requirements.txt   # Python dependencies
|-- README.md          # Project documentation
```

## Prerequisites
- Python 3.10+ (project was tested in a Python virtual environment)
- Windows, Linux, or macOS
- A modern browser (Chrome/Edge/Firefox)

## Quick Start (First-Time User)
### 1. Clone or download the repository
```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Create a virtual environment
Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run main.py
```

### 5. Open the app in your browser
Use the URL printed in terminal, usually:
```text
http://localhost:8501
```

## How to Use the App
1. Open the Image Pipeline tab.
2. Upload an image (png, jpg, jpeg, bmp, tif, tiff).
3. Select edge extractor:
   - Custom Canny
   - OpenCV Canny
4. Tune sliders:
   - low/high thresholds
   - Gaussian kernel size and sigma
   - contour minimum area
   - region-growing tolerance
5. Inspect outputs:
   - original image
   - edge map
   - contour, region-growing, and watershed results
6. In Custom Canny mode, inspect intermediate stages:
   - blurred image
   - gradient magnitude
   - non-max suppressed image
7. Review runtime/performance and sensitivity analysis.
8. Download processed outputs using the provided buttons.

## Webcam Mode (Optional)
Open the Webcam Edge Detection tab for real-time overlay.

If webcam mode does not appear/work:
- ensure streamlit-webrtc and av are installed,
- allow browser camera permissions,
- verify no other app is locking the camera.



Non-maximum suppression thins edges along local gradient direction.
Double thresholding labels strong/weak/non-edge pixels.
Hysteresis retains weak pixels connected to strong edges.

## Segmentation Methods Implemented
### Contour-Based
Edge-derived contours form region boundaries and masks.

### Region Growing
A seed-based method that expands into intensity-similar neighborhoods.

### Watershed
Marker-controlled topographic segmentation for touching/overlapping objects.

## Performance and Analysis Included
- Runtime comparison table for custom vs OpenCV Canny.
- Asymptotic complexity notes.
- Parameter sensitivity analysis (threshold changes vs edge density and runtime).

## Example Use Case Workflow
1. Capture road/walkway images in a campus block.
2. Upload images and tune edge thresholds.
3. Compare contour and watershed outputs.
4. Export maps for maintenance reporting.
5. Repeat periodically to track deterioration progression.

## Development Timeline
1. Initialized project structure and modular Python files.
2. Implemented full custom Canny stages for algorithm-level learning.
3. Added OpenCV Canny baseline for runtime comparison.
4. Implemented contour, region-growing, and watershed segmentation.
5. Built Streamlit interface for upload, tuning, visualization, and download.
6. Added webcam mode, parameter sensitivity analysis, and performance table.
7. Finalized documentation, reproducibility notes, and troubleshooting.

## Troubleshooting
### Error: No module named streamlit
Run:
```bash
pip install -r requirements.txt
```

### Port 8501 already in use
Run on another port:
```bash
streamlit run main.py --server.port 8502
```

### Webcam not working
- check browser permissions,
- close other camera-consuming apps,
- verify installation of streamlit-webrtc and av.

### App links stop working after some time
Links are active only while the Streamlit process is running. Restart the app if the terminal process stops.

## Reproducibility Notes
- Performance values vary by CPU, RAM, and image resolution.
- For publication-grade evaluation, test on standard datasets (for example BSDS500) and report precision-recall metrics.

## Running in VS Code
1. Open the project folder.
2. Select Python interpreter from your virtual environment.
3. Open terminal in the project root.
4. Activate environment.
5. Install requirements.
6. Run: streamlit run main.py

## Expected Output
The interface shows, side-by-side:
- original image,
- edge map,
- segmentation outputs,
- intermediate custom Canny stages,
- runtime and sensitivity plots.

This README is intentionally written so a new user can set up, run, and use the project without prior project context.

## Submission Checklist
- Code is version-controlled and published on GitHub.
- Setup steps are reproducible from this README.
- Required dependencies are listed in requirements.txt.
- Custom and OpenCV Canny implementations are included.
- Contour and region-based segmentation outputs are included.
- Streamlit UI supports upload, parameter tuning, and downloads.
