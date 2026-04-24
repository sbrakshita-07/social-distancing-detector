# Social Distancing Detector - Web Application

A modern web application that uses AI-powered YOLO object detection to analyze videos and detect social distancing violations in real-time.

## 🚀 Features

- **AI-Powered Detection**: Uses YOLO v3 for accurate people detection
- **Real-time Analysis**: Calculates distances between people automatically
- **Visual Feedback**: Red boxes for violations, green for safe distances
- **User Authentication**: Secure sign up and sign in system
- **Processing History**: Track all your analyzed videos
- **Modern UI**: Beautiful dark theme with smooth animations

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher** (Python 3.12 recommended)
- **pip** (Python package manager)
- **YOLO Model Files** (already included in `yolo-coco/` folder)

## 📦 Installation

### Step 1: Extract the Project

Extract the zip file to your desired location.

### Step 2: Navigate to Project Directory

Open Command Prompt (Windows) or Terminal (Mac/Linux) and navigate to the project folder:

```bash
cd path/to/social-distancing-detector
```

### Step 3: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

**Note**: If you encounter any installation errors, try:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify YOLO Model Files

Make sure the following files exist in the `yolo-coco/` folder:
- `yolov3.weights` (should be ~248 MB)
- `yolov3.cfg`
- `coco.names`

If `yolov3.weights` is missing, you'll need to download it from the official YOLO website.

## 🎯 Running the Application

### Step 1: Start the Web Server

Run the Flask application:

```bash
python app.py
```

You should see output like:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### Step 2: Open in Browser

Open your web browser and navigate to:

```
http://127.0.0.1:5000
```

or

```
http://localhost:5000
```

### Step 3: Create an Account

1. Click **"Sign up here"** on the login page
2. Fill in:
   - Username
   - Email
   - Password (minimum 6 characters)
   - Confirm Password
3. Click **"Sign Up"**

### Step 4: Upload and Process Video

1. After logging in, you'll see the dashboard
2. Click **"🎬 Upload & Process Video"** button
3. Click the upload area and select your video file (MP4, AVI, MOV, MKV, FLV, WMV)
4. Once file is selected (you'll see green checkmark), click **"Process Video"**
5. Wait for processing to complete (this may take several minutes depending on video length)
6. View results and download the processed video

## 📁 Project Structure

```
social-distancing-detector/
│
├── app.py                 # Main Flask web application
├── detector_module.py     # Core video processing module
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── configs/              # Configuration files
│   ├── config.py         # Settings (distance, frame skip, etc.)
│   └── detection.py      # YOLO detection functions
│
├── static/               # Static files (CSS, images)
│   └── style.css         # Application styling
│
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── signin.html       # Login page
│   ├── signup.html       # Registration page
│   ├── dashboard.html    # Main dashboard
│   ├── detector.html     # Video upload page
│   └── results.html      # Results display page
│
└── yolo-coco/            # YOLO model files
    ├── yolov3.weights    # Pre-trained weights (large file)
    ├── yolov3.cfg        # Model configuration
    └── coco.names        # Class labels
```

## ⚙️ Configuration

You can customize detection settings in `configs/config.py`:

- `MIN_DISTANCE`: Minimum safe distance in pixels (default: 50)
- `FRAME_SKIP`: Process every Nth frame for faster processing (default: 2)
- `FRAME_WIDTH`: Resize width for processing (default: 500)
- `USE_GPU`: Enable GPU acceleration if available (default: False)

## 🎬 Supported Video Formats

- MP4
- AVI
- MOV
- MKV
- FLV
- WMV

## 📊 Understanding Results

After processing, you'll see:

- **Total Frames**: Number of frames processed
- **Avg People per Frame**: Average number of people detected
- **Total Violations**: Number of violation pairs detected
- **Max Violations (Single Frame)**: Maximum violations in any single frame
- **Frames with Violations**: Number of frames containing violations
- **Avg Violations per Frame**: Average violations across all frames

## 🔧 Troubleshooting

### Port Already in Use

If port 5000 is already in use, edit `app.py` and change:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```
to a different port (e.g., `port=5001`)

### YOLO Weights Missing

If you get an error about missing `yolov3.weights`:
1. Download from: https://pjreddie.com/darknet/yolo/
2. Place the file in the `yolo-coco/` folder

### Slow Processing

- Increase `FRAME_SKIP` in `configs/config.py` (e.g., 3 or 4)
- Reduce `FRAME_WIDTH` in `configs/config.py` (e.g., 400)
- Close other applications to free up resources

### Import Errors

Make sure all dependencies are installed:
```bash
pip install --upgrade -r requirements.txt
```

## 📝 Notes

- Processing time depends on video length and system performance
- The database (`users.db`) is created automatically on first run
- Uploaded videos are stored in `uploads/` folder
- Processed videos are saved in `outputs/` folder
- All files are organized by user ID for multi-user support

## 🛑 Stopping the Application

Press `Ctrl + C` in the terminal/command prompt to stop the server.

## 📧 Support

If you encounter any issues:
1. Check that all dependencies are installed correctly
2. Verify YOLO model files are present
3. Ensure Python version is 3.8 or higher
4. Check the terminal/console for error messages

## 🎉 Enjoy!

You're all set! Start analyzing videos and detecting social distancing violations.

---

**Note**: This application processes videos locally on your machine. No data is sent to external servers.

