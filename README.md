Portable Neuro-Surgical Automation Platform for Hematoma Imaging and Evacuation

This repository contains the core files and scripts developed for Adam Thompson’s Master’s thesis, Portable Neuro-Surgical Automation Platform for Hematoma Imaging and Evacuation.
The project focuses on creating an automated platform capable of positioning sensors, acquiring data, and processing results for the purpose of detecting and evaluating hematomas.

Repository Contents

MATLAB / Simulink Files
Inv_Testing_v6.slx -
  Main Simulink program that executes the motion sequence between predefined points.
  Used as the central control loop coordinating system behavior.
quick_setup_v2.mat -
  Initialization file that defines and sets up required variables for the main program.
  Must be loaded prior to running Inv_Testing_v6.slx.

Python Scripts
screen_trigger_capture_v1.py - 
  Monitors the main program state and captures images from the connected camera when the system enters the hold state.
  Useful for synchronizing vision data with mechanical movement.
laser_centroid_calib.py - 
  Performs centroid detection of a laser pointer in captured images and calibrates the camera against known reference coordinates.
  Returns a list of detected centroids and their distances relative to the desired position, expressed in millimeters.

Usage
	1.	Setup
	•	Load quick_setup_v2.mat in MATLAB before running the main program.
	•	Ensure that the required toolboxes for Simulink and image acquisition are installed.
	2.	Run Main Program
	•	Open and execute Inv_Testing_v6.slx in Simulink.
	•	The program moves between specified points as defined in the setup file.
	3.	Image Capture
	•	Run screen_trigger_capture_v1.py in parallel.
	•	Images will be captured automatically when the main program reaches the hold state.
	4.	Calibration
	•	Use laser_centroid_calib.py to analyze the captured images.
	•	The script computes laser pointer centroids and outputs positional errors in millimeters.

Requirements
	•	MATLAB / Simulink (R2024b or newer recommended)
	•	Python 3.9+ with the following libraries:
	•	opencv-python
	•	numpy

Link to CAD File: https://a360.co/48yWQNR 
