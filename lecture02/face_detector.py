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
import cv2 as cv

# Load the pre-trained face and eyes detectors.
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_eye.xml")

cv.namedWindow("Face", cv.WINDOW_NORMAL)

# Create a VideoCapture object.
video = cv.VideoCapture(2)

# Loop to read the video frames.
while True:

    # Read the first frame.
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to grayscale.
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the frame.
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(300, 300))
    
    # Draw a rectangle around the faces.
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 20)

        # Detect eyes in the face.
        face = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(
            face, scaleFactor=1.1, minNeighbors=4)

        # Draw a rectangle around the eyes.
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(
                frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 10
            )

    # Show the first frame.
    cv.imshow("Frame", frame)
    if cv.waitKey(33) == ord("q"):
        break

video.release()
cv.destroyAllWindows()
