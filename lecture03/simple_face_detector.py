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

    # Detect faces in the frame.
    faces = detector(gray)

    # Loop to draw a rectangle around the detected faces.
    for face in faces:
        x0, y0 = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 20)

    # Show the first frame.
    cv.imshow("Frame", frame)
    if cv.waitKey(33) == ord("q"):
        break

video.release()
cv.destroyAllWindows()
