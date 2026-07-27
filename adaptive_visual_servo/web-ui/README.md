# SO-101 Visual Servo — React Web UI

This directory contains the modern React + Vite frontend for the SO-101 visual servoing system. The backend is served via a Flask & Socket.IO server in Python, streaming low-latency video and serial control inputs.

---

## 🚀 How to Run

### 1. Fast Launch (Production Mode)
If you just want to run the robot and the UI, simply run the Python runner script with the `--ui` flag. The Flask backend automatically serves the compiled production React assets:

```bash
# 1. Activate your virtual environment
source ../.venv/bin/activate

# 2. Run the main runner with UI flag
python run_real.py --ui
```

Now, open your web browser and navigate to:
👉 **[http://localhost:5001](http://localhost:5001)**

---

### 2. Local Development Mode (Hot Reloading)
If you want to edit the React frontend components with instant hot module replacement (HMR):

1. **Start the Flask Backend** on port 5001:
   ```bash
   python web_ui.py
   ```

2. **Start the Vite Dev Server** on port 3000 (in a separate terminal):
   ```bash
   cd web-ui
   npm run dev
   ```

3. **Open the Dev URL**:
   👉 **[http://localhost:3000](http://localhost:3000)**
   
   *Vite is configured to automatically proxy all `/api` endpoints and `/socket.io` websockets to the backend running on port 5001.*

---

## 🛠️ Step-by-Step Calibration Guide

Once the browser UI is loaded, follow these steps to register your workspace:

### Step 1: Connect to Camera & Arm
1. Click **Scan** to search for connected cameras.
2. Select your camera index and check your serial port (e.g. `/dev/ttyACM0` or `/dev/cu.usbmodem1101`).
3. Click **Connect Arm** to open the serial link and video stream.

### Step 2: Camera Intrinsics
Before locating markers in 3D, the system needs camera parameters (focal length, principal points, and distortion coefficients):
* **Method A (Accurate)**: Position a ChArUco board under the camera, click **Collect Frame** in at least 5 different angles, and click **Calibrate**.
* **Method B (Instant Estimate)**: If you don't have a ChArUco board handy, click **Estimate**. The system calculates fallback values based on your camera resolution.

### Step 3: Workspace Extrinsics (PnP)
1. Verify that your four workspace markers (IDs 0, 1, 2, and 3) are visible at the table edges.
2. Under **Workspace Extrinsics**:
   * Set your workspace marker size (default `0.033 m` / `3.3 cm`).
   * Enter your physical table dimensions (Width and Depth, e.g. `0.63 m` x `0.60 m`).
3. Click **Calibrate Workspace**. The system will calculate the camera's relative 3D pose, and show the RMS reprojection error (ideally < 1.0px).

---

## 🎮 Controlling the Arm

* **Reset Arm**: Smoothly interpolates the joint motor coordinates back to the home posture.
* **Circle / Heart**: Draws vertical circular or heart trajectories in front of the robot at a high 40Hz resolution.
* **Target Clicks**:
  1. Click **Set Pick** and then click a point on the table surface in the video stream. The system will backproject the pixel coordinate to a 3D workspace location (`Z=0`).
  2. Click **Set Drop** and select where the robot should place the object.
  3. Click **Run Episode** to execute a visual servo task using active ArUco marker tracking!
