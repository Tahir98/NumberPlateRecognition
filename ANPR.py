
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import imutils
import cv2 
from matplotlib import pyplot as plt
import numpy as np
import easyocr


image = cv2.imread("image3.jpg")
width,height,channel = image.shape
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

fig = plt.figure(figsize=(10, 7))

# setting values to rows and column variables
rows = 2
columns = 2

fig.add_subplot(rows, columns, 1)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.axis('on')
plt.title("Normal")

fig.add_subplot(rows, columns, 2)
plt.imshow(cv2.cvtColor(gray,cv2.COLOR_BGR2RGB))
plt.axis('on')
plt.title("Gray-Scale")

#Filter and edge detection
filtered = cv2.bilateralFilter(gray, 9, 75, 75)#noise reduction
edged = cv2.Canny(filtered,30,200) #edge dectection

fig.add_subplot(rows, columns, 3)
plt.imshow(cv2.cvtColor(edged,cv2.COLOR_BGR2RGB))
plt.axis('on')
plt.title("Edged")

#Find contours
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours,key=cv2.contourArea,reverse=True)[:5]

counter = 0
length = 0
cX,cY,cW,cH = 0,0,0,0
text = ""

for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        x,y,w,h =cv2.boundingRect(approx)
        cropped = gray[y:y+h,x:x+w]

        #EasyOCR
        reader = easyocr.Reader(["tr"])
        result = reader.readtext(cropped)

        if len(result) > 0 and len(result[0][1]) > length:
            counter = counter + 1
            length = len(result)
            print(result)
            text = result[0][1]
            cX,cY,cW,cH = x,y,w,h
            print(text)

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = float(height)/600.0
print(text)
print(len(text))

if counter == 0:
    cv2.putText(image,"Plate Not Found",(5,20),font,fontScale,(0,0,255),1,cv2.LINE_AA)   
else:
    cv2.rectangle(image, (cX,cY),(cX+cW,cY+cH),(0,255,0),thickness=6)
    cv2.putText(image,text,(50,50),font,fontScale,(0,0,255),2,cv2.LINE_AA)

fig.add_subplot(rows, columns, 4)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.axis('on')
plt.title("Result")
plt.show()








    