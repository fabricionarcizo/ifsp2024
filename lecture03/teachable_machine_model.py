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
import numpy as np

from keras.models import load_model

# Disable scientific notation for clarity.
np.set_printoptions(suppress=True)

# Load the model.
model = load_model("./data/models/cube-colors/keras_model.h5", compile=False)

# Load the labels.
class_names = open("./data/models/cube-colors/labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer.
camera = cv.VideoCapture(0)

while True:

    # Grab the webcamera's image.
    ret, image = camera.read()
    if not ret:
        break

    # Resize the raw image into (224-height,224-width) pixels.
    resized = cv.resize(image, (224, 224), interpolation=cv.INTER_AREA)
    resized = cv.cvtColor(resized, cv.COLOR_BGR2RGB)

    # Make the image a numpy array and reshape it to the models input shape.
    resized = np.asarray(resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    resized = (resized / 127.5) - 1

    # Predicts the model
    prediction = model.predict(resized)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    cv.putText(image, "Class: " + class_name[2:], (10, 50),
                cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # Show the image in a window
    cv.imshow("Webcam Image", image)
    if cv.waitKey(33) == ord("q"):
        break

camera.release()
cv.destroyAllWindows()
