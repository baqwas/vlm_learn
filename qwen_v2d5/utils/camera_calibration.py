#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
camera_calibration.py
This script performs camera calibration using a set of images of a chessboard pattern.
It uses OpenCV to find the chessboard corners, calculate the camera matrix,
and distortion coefficients.
Usage:
    python camera_calibration.py
Dependencies:
    python -m pip install opencv-python numpy glob2
Notes:
    - Ensure you have a folder named 'calibration_images' with chessboard images.
    - The chessboard pattern should be a standard one with known dimensions.
    - The script assumes the chessboard is flat and on the Z=0 plane.
MIT License

Copyright (c) 2023 ParkCircus Productions

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import cv2 as cv
import glob

# --- 1. Define the dimensions of the chessboard pattern ---
# You need to know the number of inner corners of your chessboard.
# For a standard 9x6 chessboard, there are 8x5 inner corners.
# chessboard_size = (8, 5)
# chessboard_size = (7, 7)
# chessboard_size = (6, 4)
chessboard_size = (14, 9)
# --- 2. Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ... ---
# These are the 3D coordinates of the corners in the real world.
# We assume the chessboard is on the Z=0 plane.
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# --- 3. Prepare lists to store object points and image points ---
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in the image plane

# --- 4. Read the calibration images ---
# The glob module finds all pathnames matching a specified pattern.
# images = glob.glob('/home/reza/Videos/FTC/calibrate/images/VisionPortal*.png')
images = glob.glob('/home/reza/Videos/FTC/calibrate/images/2024/Board2/VisionPortal*.png')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

    # If corners are found, add object points and image points
    if ret:
        objpoints.append(objp)
        # Refine the corner locations for higher accuracy
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        # Optional: Draw and display the corners
        # cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(500)

cv.destroyAllWindows()

# --- 5. Perform the calibration ---
# This is the core function that calculates the camera matrix, distortion coefficients,
# rotation and translation vectors.
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# --- 6. Print the results ---
if ret:
    print("Calibration successful!")
    print("\nCamera Matrix (K):\n", camera_matrix)
    print("\nDistortion Coefficients (D):\n", dist_coeffs)

    # --- 7. (Optional) Re-projection Error ---
    # A low re-projection error indicates a good calibration.
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("\nMean Re-projection Error: {:.2f} pixels".format(mean_error / len(objpoints)))
else:
    print("Calibration failed.")