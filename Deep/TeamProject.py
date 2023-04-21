##기본 라이브러리
import numpy as np
import pandas as pd
##시각화 관련 라이브러리
import matplotlib.pyplot as plt
import cv2 as cv
## 딥러닝 라이브러리
import tensorflow as tf
from tensorflow import keras
## Crawling 라이브러리
from selenium import webdriver

# ## 얼굴인식
# ### opencv 라이브러리
# faceCascade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')
# face_img = cv.imread('./img/1.jpg')
# frameGray = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
# faces = faceCascade.detectMultiScale(frameGray)
# for face in faces:
#     x1, y1, w, h = face 
#     x2 = x1 + w 
#     y2 = y1 + h
# ###Dlib 라이브러리
# #FaceDetector = dlib.get_frontal_face_detector()
# #face = FaceDetector(face_img)
# #for face in faces:
#  #     x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
# ### 피부색상 추출
# #RGB ->YCbCr 변환
# face_img_ycrcb = cv.cvtColor(face_img, cv.COLOR_BGR2YCrCb)
# #YCbCr 마스크 생성
# lower = np.array([0,133,77], dtype = np.uint8)
# upper = np.array([255,173,127], dtype = np.uint8)
# skin_msk = cv.inRange(face_img_ycrcb, lower, upper)
# #생성한 마스크 원본 이미지에 입힘
# skin = cv.bitwise_and(face_img, face_img, mask = skin_msk)
# cv_imshow(skin)


# #피부영역 분리

# #CNN 모델

# #검증