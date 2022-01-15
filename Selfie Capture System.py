
import numpy as np
import cv2

import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

i = 3

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        font = cv2.FONT_HERSHEY_SIMPLEX
        if i != -1:
            cv2.putText(img,'Face detected.' + str(i),(x-10,y-10), font, 0.5, (11,255,255), 1, cv2.LINE_AA)
        else :
            cv2.putText(img,'Face detected. Image Saved.',(x-10,y-10), font, 0.5, (11,255,255), 1, cv2.LINE_AA)
        if i != -1 :
            
            if i == 0:
                print("picture saved")
                cv2.imwrite('photo'+str(i)+'.jpg',img)
            print(i)

            time.sleep(1)
            i-=1

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            

    cv2.imshow('img',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()