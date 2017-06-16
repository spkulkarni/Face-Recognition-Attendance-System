import numpy as np
import cv2,os
from PIL import Image
from PIL import *
from pymongo import MongoClient
from datetime import datetime

client = MongoClient()
db = client.test
res = db.studentdetail3.insert_one(
    { "student":
         [
            {
                "Name": "Renukapriya K",
                "USN": "1MS14CS102",
                "atten":0,
                "key":1
            },
            {
                "Name": "S M Leelavathi",
                "USN": "1MS14CS105",
                "atten":0,
                "key":2
            },
            {
                "Name": "Sharvani Kulkarni",
                "USN": "1MS14CS115",
                "atten":0,
                "key":3

            },
            {
                 "Name": "Spoorthy Hathwar",
                "USN": "1MS14CS123",
                "atten":0,
                "key":4
            },
             {
                "Name": "Thanushree J",
                "USN": "1MS14CS133",
                "atten":0,
                "key": 5
            }

        ],

    }
)


faceCascade = cv2.CascadeClassifier('/home/avvi/PycharmProjects/ai_project/haarcascade_frontalface_default.xml')
recognizer = cv2.face.createLBPHFaceRecognizer()
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.gif')]
    images=[]
    labels=[]
    for image_path in image_paths:
         image_pil = Image.open(image_path).convert('L')
         image = np.array(image_pil, 'uint8')
         nbr = int(os.path.split(image_path)[1].split(".")[0].replace("photo", ""))
         faces = faceCascade.detectMultiScale(image)
         for (x, y, w, h) in faces:
             images.append(image[y: y + h, x: x + w])
             labels.append(nbr)
             cv2.resizeWindow("Adding faces to traning set...",200,200)
             cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
             cv2.waitKey(100)
    return images, labels
path='/home/avvi/PycharmProjects/ai_project/yalefaces/smj'
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()
recognizer.train(images, np.array(labels))

camera = cv2.VideoCapture(0)

while(camera.isOpened()):
    ret, frame = camera.read()
    if ret==True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            #eyes = eye_cascade.detectMultiScale(roi_gray)
        if cv2.waitKey(1) & 0xFF == ord('e'):
            cv2.imwrite("/home/avvi/PycharmProjects/ai_project/yalefaces/smj1/photorec.png", frame)
        cv2.imshow('window-name',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
path1='/home/avvi/PycharmProjects/ai_project/yalefaces/smj1'

image_paths1 = [os.path.join(path1, f) for f in os.listdir(path1) if f.endswith('.png')]
image_paths2 = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.gif')]
i=0
a=[]
for image_path2 in image_paths2:
    nbr_actual = int(os.path.split(image_path2)[1].split(".")[0].replace("photo", ""))
    a.append(nbr_actual)
    #qprint(a)

j=0

for image_path1 in image_paths1:
    predict_image_pil = Image.open(image_path1).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)

    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
        #nbr_actual = int(os.path.split(image_path2)[1].split(".")[0].replace("photo", ""))
        for j in range(len(a)):
            if a[j] == nbr_predicted:
                #print(a[j])
                print("{} is Correctly Recognized with confidence {}".format(a[j], conf))
                if db.studentdetail3.find_one({"student.key":a[j]}):
                    k=db.studentdetail3.find_one({"student.key":a[j]},{"student.$":1 })
                    m=k['student']
                    print(k)

                    for v in m:
                        if 'atten' in v:
                            n=v['atten']
                #n=m[0]['atten']
                            n=n+1

                            resup=db.studentdetail3.update_one({'student.key':a[j]},{'$set':{'student.$.atten':n}})
                            print(resup.matched_count)
                            k=db.studentdetail3.find_one({"student.key":a[j]},{"student.$":1 })
                            m=k['student']
                            print(k)
            #j=j+1
            else:
                print()
                #print("{} is Incorrectly Recognized as {}".format(a[j], nbr_predicted))
            #j=j+1
        cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
        cv2.waitKey(1000)


res.inserted_id
cursor = db.studentdetail3.find()

