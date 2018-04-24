import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC
import sys

emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]  # Emotion list
# emotions = ["contempt", "disgust", "fear", "happiness", "neutral", "surprise"]  # Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Or set this to whatever you named the downloaded file
clf = SVC(kernel='linear', probability=True, tol=1e-3)  # , verbose = True) #Set the classifier as a support vector machines with polynomial kernel

data = {}  # Make dictionary for all values


# data['landmarks_vectorised'] = []

def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset\\%s\\*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return training, prediction


def get_landmarks(image):
    detections = detector(image, 1)
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))

        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" % emotion)
        training, prediction = get_files(emotion)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised'])  # append image array to training data list
                training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


accur_lin = []
for i in range(0, 10):# 10):
    print("Making sets %s" % i)  # Make sets by random sampling 80/20%
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    npar_train = np.array(training_data)  # Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("training SVM linear %s" % i)  # train SVM
    clf.fit(npar_train, training_labels)

    print("getting accuracies %s" % i)  # Use score() function to get accuracy
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print
    "linear: ", pred_lin
    accur_lin.append(pred_lin)  # Store accuracy in a list

print("Mean value lin svm: %s" % np.mean(accur_lin))  # FGet mean accuracy of the 10 runs

def most_common(lst):
    return max(set(lst), key=lst.count)

video_capture = cv2.VideoCapture(0) #Webcam object
lastPred = 1
count = 0; maxCount = 9
predictionsList = []
while True:
    prediction_data = []
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)


    get_landmarks(clahe_image)
    prediction_data.append(data['landmarks_vectorised'])
    npar_pred = np.array(prediction_data)

    predictionsList.append(clf.predict(npar_pred)[0])

    if count == maxCount:
        count = 0
        # currPred = np.argmax(np.bincount(clf.predict(npar_pred)))
        currPred = np.argmax(np.bincount(predictionsList))


        # if lastPred != currPred:
            # print("Prediction: " + emotions[currPred])
        # print("Prediction: " + emotions[currPred] + " || Percentages: " + emotions[0] + " " + str(predictionsList.count(0)/(maxCount + 1)*100) + " %, " + emotions[1] + " " + str(predictionsList.count(1)/(maxCount + 1)*100) +
        #       " %, " + emotions[2] + " " + str(predictionsList.count(2)/(maxCount + 1)*100) + " %, " + emotions[3] + " " + str(predictionsList.count(3)/(maxCount + 1)*100) + " %, " +
        #       emotions[4] + " " + str(predictionsList.count(4)/(maxCount + 1)*100) + " %, " + emotions[5] + " " + str(predictionsList.count(5)/(maxCount + 1)*100) + " %, " +
        #       emotions[6] + " " + str(predictionsList.count(6)/(maxCount + 1)*100) + " %, " + emotions[7] + " " + str(predictionsList.count(7)/(maxCount + 1)*100))

        print("Prediction: " + emotions[currPred] + " || Percentages: ", end="")
        for i in range(len(emotions) - 1):
            print(emotions[i] + " " + str(predictionsList.count(i)/(maxCount + 1)*100) + " %, ", end="")

        print(emotions[len(emotions)-1] + " " + str(predictionsList.count(len(emotions)-1) / (maxCount + 1) * 100))

        lastPred = currPred

        predictionsList = []
        
    count += 1
    prediction_data = []

    detections = detector(clahe_image, 1)  # Detect the faces in the image

    for k, d in enumerate(detections):  # For each detected face

        shape = predictor(clahe_image, d)  # Get coordinates
        for i in range(1, 68):  # There are 68 landmark points on each face
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), thickness=2)  # For each point, draw a red circle with thickness2 on the original frame

    cv2.imshow("image", frame)  # Display the frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit program when the user presses 'q'
        break