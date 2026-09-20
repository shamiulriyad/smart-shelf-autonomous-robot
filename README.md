# ArUco Marker Detection

## Warehouse pick-and-place prototype

A computer-vision prototype of a robot that picks a product (identified by its ArUco marker), looks up its shelf, and places it there. **The robot is simulated** and the room map is predefined. The webcam supplies marker detections only; it does **not** localise the robot, and the simulated (x, y) position is not derived from the camera.

```
Camera -> aruco/ (detect, pose) -> products/ (marker -> product -> shelf ID) -> shelves/ (shelf -> marker, map position)
       -> navigation/ (map, A* planner) -> robot/ (state machine drives Robot; SimulatedRobot logs actions)
```

| Path | Role |
|---|---|
| `marker_detection_system.py` | Original detector (unchanged, reused by `aruco/detector.py`) |
| `aruco/` | `detector.py` structured detections, `pose_estimator.py` solvePnP pose, `calibration.py` chessboard calibration, `perception.py` camera + overlay facade |
| `camera/camera_source.py` | `CameraSource` interface, `WebcamSource`, `SyntheticCameraSource` (add a Pi camera here later) |
| `config/locations.json` | Single source of truth: marker types (dictionary, ID range, physical size), products, shelves, room map |
| `config/loader.py` | Loads and validates the config |
| `products/`, `shelves/` | Product and shelf lookups |
| `navigation/` | Occupancy-grid map and A* path planner |
| `robot/robot_controller.py` | `Robot` interface + `SimulatedRobot` |
| `robot/state_machine.py` | SEARCH_PRODUCT -> ... -> DONE, with RECOVERY and FAILED |
| `main.py` | CLI |
| `tools/generate_markers.py` | Marker PNGs generated from the config's dictionary |
| `tools/generate_marker_pdf.py` | Printable A4 PDF of labelled markers (products and/or shelves) at exact physical size |
| `tools/verify_markers.py` | Checks that generated PNGs (or a PDF) are detected by the project's detector |

### Running

```bash
pip install -r requirements.txt                       # OpenCV 4.7+ and NumPy
python tools/generate_markers.py                      # print the markers from ./markers at the sizes it reports
python tools/generate_marker_pdf.py --all             # printable A4 PDF -> markers/pdf/all_markers.pdf
python tools/verify_markers.py                        # detect every marker PNG in ./markers
python main.py --show-map                             # check the map and planned paths
python main.py --pick 102                             # webcam demo ("Product B"); Q or ESC quits
python main.py --pick 102 --synthetic                 # same flow on rendered markers, no webcam
python -m unittest discover -s tests -t .             # tests
```

For `--pick 102`: hold marker 102 in front of the camera (SEARCH/DETECT/VERIFY/PICK), then carry the camera to shelf marker 203 and show it (DETECT_SHELF/ALIGN_WITH_SHELF), then **walk up to the shelf** until the marker measures within the stand-off (APPROACH_SHELF); the robot then does MOVE_TO_PLACEMENT_POSITION, PLACE_PRODUCT and VERIFY_PLACEMENT. Detecting the shelf marker only confirms the shelf was found, and seeing it from across the room is not reaching it: APPROACH_SHELF measures the camera-to-marker distance on every frame and prints how much closer to go, and the product stays in the gripper until that gap is closed. Other product or shelf markers in view are ignored and reported as ignored. Losing a marker sends the machine to RECOVERY, which retries up to 3 times before FAILED.

With a real webcam the search is driven by a person, not by the simulated sweep, so each search waits on the wall clock (`MachineSettings.for_live_camera`: up to 180 s per search) and prints a progress line every few seconds while it waits — walking to the shelf is normal, not a failure. `--synthetic` keeps the frame-count budgets, because there the simulated robot aims the camera itself.

The same split applies to the approach. The simulated robot drives its own measured gap, creeping forward and re-measuring after each step, and never past the placement point the map allows. A person carrying the camera is the drive train, so the machine only measures and reports; nothing but the measurement can end APPROACH_SHELF. Without a calibration the distance is an apparent-size estimate, logged as such on every line — run `--calibrate` for a real one, and make sure the printed marker really is the size `config/locations.json` claims, because the estimate scales directly with it.

### Configuration

Edit `config/locations.json`. `alignment.standoff_cm` is how far in front of the shelf face the robot places from, and `alignment.approach_tolerance_cm` is how much further than that a measured distance may still read and count as arrived (default 30 cm and 10 cm, so APPROACH_SHELF accepts 40 cm or closer) — loosen it if an uncalibrated camera makes the band hard to hit by hand. Product markers (IDs 100-199, 5 cm) and shelf markers (IDs 200-299, 18.7 cm) are separate marker types with their own dictionary, ID range and size; the loader rejects overlaps, unknown shelves, out-of-range IDs and unreachable approach points. Change the physical sizes here if you print different markers. Use `tools/generate_markers.py` rather than markers from online generators, whose dictionary may not match.

### Printing markers

```bash
python tools/generate_marker_pdf.py --all                                # markers/pdf/all_markers.pdf
python tools/generate_marker_pdf.py --products                           # markers/pdf/product_markers.pdf
python tools/generate_marker_pdf.py --shelves                            # markers/pdf/shelf_markers.pdf
python tools/generate_marker_pdf.py --product-ids 101 102 --shelf-ids 203   # markers/pdf/all_markers_selected.pdf
python tools/generate_marker_pdf.py --all --product-size-cm 4 --shelf-size-cm 10   # override the config sizes
python tools/verify_markers.py --pdf markers/pdf/all_markers.pdf         # detect every marker in a PDF
```

Products and shelves, their names and marker IDs, dictionary and default sizes all come from `config/locations.json`; requested IDs must be configured and inside the type's ID range. The PDF is only written after the project's detector has found every marker in the rendered pages, at the requested size. **Print at 100% / Actual Size, never Fit to Page**, and check the 10 cm scale bar on the page with a ruler. The size is that of the black square; the label's white border must stay around it. An 18.7 cm marker nearly fills the A4 width, so its white border is only about 11 mm on the paper: do not trim it.

### Camera calibration and pose

Until a calibration exists, the camera is **uncalibrated**: no position or rotation is reported for any marker. Only image-space cues (offset from image centre, apparent size, and a rough distance guess) are shown, and alignment is labelled non-metric. To calibrate:

```bash
python main.py --calibrate --square-cm 2.5            # measure your printed chessboard squares with a ruler
```

Show a chessboard (default 7x6 inner corners, `--board COLSxROWS` to change) at varied distances and tilts; SPACE captures, ENTER finishes (10+ views, ~20 recommended). This writes `config/camera_calibration.json`, which is per camera and per resolution: pose is disabled if the live frame size differs (use `--width/--height` in both steps). Aim for an RMS reprojection error under 0.5 px. With a calibration, each marker gets `rvec`/`tvec`, relative x/y/z (cm), bearing and yaw. Expect errors to grow with distance, small markers and steep viewing angles.

### Limits of the prototype

Robot position is simulated; motors, gripper and real localisation are not implemented. VERIFY_PLACEMENT checks the simulated gripper state only. Marker detection needs decent lighting and a white border around each printed marker.

## Installation 🚀

### Requirements 📋

Before diving into the installation process of the ArUco Marker Detector, ensure that your system meets the following requirements:

- **Python 3.6+**: The ArUco Marker Detector is compatible with Python 3.6 and higher versions. You can download the latest version of Python from the official website [here](https://www.python.org/downloads/).
- **OpenCV**: OpenCV (Open Source Computer Vision Library) is a vital dependency for image processing and computer vision tasks. You can install OpenCV using pip, the Python package installer.
- **NumPy**: NumPy is a fundamental package for scientific computing with Python and is required for array manipulation and numerical operations. It's essential to have NumPy installed to use the ArUco Marker Detector effectively.

### Steps 🛠️

Follow these detailed steps to install the ArUco Marker Detector on your machine:

1. **Clone the Repository**: Begin your journey by cloning this repository to your local machine using Git. Open your terminal or command prompt and execute the following command:
   ```bash
   git clone https://github.com/Rishit-katiyar/ArUcoMarkerDetector.git
   ```

2. **Navigate to Project Directory**: Once you've successfully cloned the repository, navigate to the project directory using the `cd` command:
   ```bash
   cd ArUcoMarkerDetector
   ```

3. **Install Python**: If Python is not already installed on your system, you can download and install it from the official Python website [here](https://www.python.org/downloads/). Follow the installation instructions provided for your operating system.

4. **Install OpenCV**: OpenCV can be installed using pip, the Python package installer. Execute the following command to install the OpenCV library:
   ```bash
   pip install opencv-python
   ```

5. **Install NumPy**: Similarly, NumPy can be installed using pip. Execute the following command to install the NumPy package:
   ```bash
   pip install numpy
   ```

6. **Verify Installation**: After installing the dependencies, it's essential to verify the installation to ensure everything is set up correctly. You can do this by running a simple Python script that imports the required libraries. Create a new Python script (e.g., `verify_installation.py`) and add the following code:
   ```python
   import cv2
   import numpy as np

   print("OpenCV version:", cv2.__version__)
   print("NumPy version:", np.__version__)
   ```

   Save the script and execute it using the Python interpreter. If the installation was successful, you should see the versions of OpenCV and NumPy printed to the console.

7. **Congratulations!**: 🎉 You have successfully installed the ArUco Marker Detector and its dependencies on your system. You're now ready to delve into the exciting world of detecting, tracking, and visualizing ArUco markers in images or videos.

8. **Additional Resources**: For more information on OpenCV and NumPy, you can refer to their official documentation:
   - [OpenCV Documentation](https://docs.opencv.org/)
   - [NumPy Documentation](https://numpy.org/doc/)

9. **Troubleshooting**: If you encounter any issues during the installation process, don't panic! We've got you covered. Check out the troubleshooting section below for common solutions to potential problems.

10. **Feedback and Contributions**: 🤝 We value your feedback and welcome contributions from the community. If you have suggestions for improvement or want to contribute to the project, please open an issue or submit a pull request on GitHub.

11. **License**: This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

12. **Acknowledgements**: Special thanks to the developers and contributors of OpenCV and NumPy for their outstanding work and contributions to the fields of computer vision and scientific computing.

13. **Stay Updated**: Don't forget to star the repository on GitHub and follow us for updates and announcements! ⭐️

14. **Happy Coding!**: 🚀 We hope you find the ArUco Marker Detector useful for your projects and experiments. Happy coding! 😊
