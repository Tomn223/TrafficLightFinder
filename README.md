# TrafficLightFinder
Traffic light classification of LISA traffic light dataset using OpenCV

## Problem

## Solution & Rationale
I sourced and formatted images from the LISA traffic light dataset, then used OpenCV to train a cascade classifier specifically for detecting traffic lights. 

### Image Data
The LISA traffic light dataset had images 13 different clips of daytime driving, each with their own CSV annotation file which contained bounding box coordinates for (nearly) all traffic lights present in the images. I initally only used 1 of these clips, but ended up using 3 clips because I needed more data to improve the model.

### Training a Cascade Classifier with OpenCV
Like any machine learning approach, the classifier required lots of data to train. Specifically, to run OpenCV's train_cascade applicaton, I needed to pass many positive images (images with traffic lights) with annotations and negative images. The annotations required a different format than what the LISA dataset provided, so I had to create this annotation .dat file. Here is an example annotaion:

```
img/img2.jpg 2 100 200 50 50 50 30 25 25
```

From left to right, it requires the file location of the image, number of traffic lights in the image, then their bounding box parameters (Upper left x, Upper left y, width, height) after. The negatives just require a .txt file with the negative file locations. After you have these text files, you can run the following command to create positive samples .vec file, which is needed to run the train_cascade executable.

```bash
(path to open cv download)/opencv/build/x64/vc15/bin/opencv_createsamples -info image_descriptor/positive.dat -w 20 -h 30 -num 1000 -vec image_descriptor/positive.vec
```

You can see the positive.dat annotation file, the width and height of output objects, the number of samples created, and the .vec file created. Now, you're ready to train the model, using the following executable:

```bash
(path to open cv download)/opencv/build/x64/vc15/bin/opencv_traincascade -data cascade/ -vec image_descriptor/positive.vec -bg image_descriptor/negative.txt -w 20 -h 30 -numPos 500 -numNeg 1600 -numStages 10 -minHitRate 0.998 -maxFalseAlarmRate 0.3
```

Here, I've specified the folder the trained classifier will go (cascade/), the positive sample .vec file created from the previous executable, the negative file, same width and height, the number of positive and negative samples used each stage, and the number of training stages. The minHitRate and maxFalseAlarmRate are arguments that I passed to raise their threshold. This seemed to give me a better model.


After the training, you will be left with a classifier specified in a .xml file, which can be used to create a classifier and detect traffic lights in images as such:

```
tl_cascade_classifier = cv2.CascadeClassifier("cascade/cascade.xml")
boxes = tl_cascade_classifier.detectMultiScale(img)
```

The detectMultiScale returns a list of bounding boxes for the detected traffic lights.

### HAAR Cascade Classifier
The classifier uses simple HAAR features to build object classifiers. We try each haar feature and try them for every location and size. Most of these filters end up being irrelevant, but through Adaboost we can extract the most useful features (the ones that minimize error). 


This still may leave us with many features which are useless at most locations in a window. That's where the cascade aspect comes into play, as these features are grouped into stages designed to weed out non traffic lights. Candidate objects are thrown out if they don't reach a certain matching threshold in a stage. This rapidly improves the efficiency of a cascade classifier.

### Color detection
After I detect the traffic lights, I'm passing the cropped image to a color detection function. This function converts the image to HSV, then checks if each pixel is within certain bounds corresponding to red and yellow. If enough red/yellow/green are detected, it will return a string with that color.
## Hardships/Improvements
- Preprocess images
- Need more data, better variety (I only trained my classifier on images with black traffic lights, so it's failing on the yellow traffic light images)
- The color detection function isn't working very well
- Annoying data layout, model training format
- Executables to train/create samples are hard to work with


## Links
OpenCV tutorials:
- https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
- https://docs.opencv.org/4.x/dc/d88/tutorial_traincascade.html


LISA traffic light dataset:
- https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset


Preprocessing Techniques (Hopefully added, if time permits):
- https://medium.com/@kenan.r.alkiek/https-medium-com-kenan-r-alkiek-traffic-light-recognition-505d6ab913b1


Other:
- https://en.wikipedia.org/wiki/Cascading_classifiers
- https://www.youtube.com/watch?v=XrCAvs9AePM
