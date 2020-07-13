from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

#set local or full pathname for the haarcascades and vgg.h5 files
face_classifier = cv2.CascadeClassifier('/Users/alex/Desktop/emotion_detection-master/haarcascade_frontalface_default.xml')
classifier = load_model('/Users/alex/Desktop/emotion_detection-master/Emotion_little_vgg_epoch25.h5')

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
cap = cv2.VideoCapture(0)

# set variable for falling check
arrHeights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
count = 0
fallen = False


while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # Falling check
        arrHeights.append(y)  # append array of y coordinates for each face
        size = len(arrHeights)

        # loop through the last 10 x coordinates to check for downward trend
        for s in range(size - 11, size - 1):
            if arrHeights[s] > arrHeights[s - 1]:
                count = count + 1
            else:
                count = 0
                fallen = False
                # if user comes back into screen, the fallen message should disappear

            # make sure that the fall is large/fast enough and the user wan't just moving their head downward
            if count >= 6 and arrHeights[s] - arrHeights[s - 5] >= 175:
                fallen = True
                break

        # draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        # rect,face,image = face_detector(frame)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            print(class_labels)
            print(preds)
            print(label)
            print()
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    if fallen:
        # print("Fallen")
        cv2.putText(frame, "FALLEN", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
