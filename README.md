# Application for Visually Challenged
This is a Python script that detects objects in real-time video using OpenCV's Deep Neural Network (DNN) module and the pre-trained MobileNet SSD model, and also recognizes text in the detected objects using Pytesseract.

## Requirements
Python 3.x
OpenCV
Pytesseract

## Usage
Clone this repository to your local machine.
Install the required libraries using pip: pip install opencv-python pytesseract.
Connect a camera to your computer.
Run the script using the following command: python object_detection.py.

## Functionality
The script captures video frames from the camera and uses the MobileNet SSD model to detect objects in the frames.
The script identifies objects that are located in the middle of the frame (within 100 pixels of the center) and draws a bounding box around them.
If the user presses the 'r' key, the script extracts the text within the bounding box of the object using Pytesseract and displays it on the console.
If the user presses the 'd' key, the script calculates the distance between the object and the camera and speaks it using the Pyttsx3 text-to-speech library.
If the user presses the 'h' key, the script speaks the names of all the objects currently located in the middle of the frame, and then clears the list.
The script stops running when the user presses the 'q' key or when the program is manually interrupted.

## License
This project is licensed under the terms of the MIT license.
