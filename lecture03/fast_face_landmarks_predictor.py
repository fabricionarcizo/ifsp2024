#
# MIT License
#
# Copyright (c) 2024 Fabricio Batista Narcizo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#/

# Import the necessary packages.
import dlib # To install this package: pip install dlib
import cv2 as cv

# Load the face detector.
detector = dlib.get_frontal_face_detector()

# Load the face landmarks predictor.
predictor = dlib.shape_predictor(
    "./data/dlib/shape_predictor_68_face_landmarks.dat")

# Create a VideoCapture object.
video = cv.VideoCapture(0)

# Loop to read the video frames.
while True:

    # Read the first frame.
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to grayscale.
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Resize the frame.
    level = 2
    for i in range(level):
        gray = cv.resize(gray, (0, 0), fx=0.5, fy=0.5)

    # Detect faces in the frame.
    faces = detector(gray)

    # Loop to draw a rectangle around the detected faces.
    for face in faces:
        
        # Predict the landmarks of the face.
        landmarks = predictor(gray, face)

        # Loop to draw a circle around each landmark.
        for i in range(68):
            x = landmarks.part(i).x * (2 ** level)
            y = landmarks.part(i).y * (2 ** level)
            cv.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Show the first frame.
    cv.imshow("Frame", frame)
    if cv.waitKey(33) == ord("q"):
        break

video.release()
cv.destroyAllWindows()
