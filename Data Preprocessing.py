

import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt

female_bath=glob('D://BISM/mlops/2_Train_FaceRecognition_with_ML/data/data/female/*.jpg')
male_bath=glob('D://BISM/mlops/2_Train_FaceRecognition_with_ML/data/data/male/*.jpg')


print(len(male_bath))
print(len(female_bath))


#read all images and convert them to RGB

for i in range(len(female_bath)):
    try:

        img=cv2.imread(female_bath[i])
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        #apply harcascade classifier

        haar_cascade=cv2.CascadeClassifier('D://BISM/mlops/1_OpenCV/data/haarcascade_frontalface_default.xml')
        gray_img=cv2.cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces_list=haar_cascade.detectMultiScale(gray_img , 1.5 ,5 )



        for x,y,w,h in faces_list:



        #crop faces
            roi=img_rgb[y:y+h , x:x+w]
            roi_rgb=cv2.cvtColor(roi,cv2.cv2.COLOR_BGR2RGB)


        #save the image
            cv2.imwrite(f'D://BISM/mlops/cropped_images_data/female/female{i}.jpg',roi_rgb)
            print("image is successfully processed ")
    except:
        print("unable to process")

#for male
for i in range(len(male_bath)):
    try:

        img=cv2.imread(male_bath[i])
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        #apply harcascade classifier

        haar_cascade=cv2.CascadeClassifier('D://BISM/mlops/1_OpenCV/data/haarcascade_frontalface_default.xml')
        gray_img=cv2.cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces_list=haar_cascade.detectMultiScale(gray_img , 1.5 ,5 )



        for x,y,w,h in faces_list:



        #crop faces
            roi=img_rgb[y:y+h , x:x+w]
            roi_rgb=cv2.cvtColor(roi,cv2.cv2.COLOR_BGR2RGB)


        #save the image
            cv2.imwrite(f'D://BISM/mlops/cropped_images_data/male/male{i}.jpg',roi_rgb)
            print("image is successfully processed ")
    except:
        print("unable to process")

#######################################################################################################################

#Data exploration and analysis:

# 1-distripution of male-female:
    # *bie chart
    # *bar chart

# 2-what is the distribution size of all images:
    # *histogram
    # *box plot
    # *split by gender

# 3- make descision what is the size of width and hieght

# 4- removing the images with less choosen  size

