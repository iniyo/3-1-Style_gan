import numpy as np
import cv2, glob
import matplotlib.pyplot as plt
import keras
train_images, test_images, train_labels, test_labels = np.load('C:\\Users\\iniyo\\Desktop\\face_Datasets\\learning_dataset=128.npy', allow_pickle = True) 

de_sk=[]
# 사용자의 dir
user_face_dir = 'C:\\Users\\iniyo\\Desktop\\face_Datasets\\test_images\\'
user_face_img_list = glob.glob(user_face_dir+"/*.jpg") # 해당 폴더 내의 jpg형식의 모든 파일을 list형식으로 가져옴
# 모델 가져오기
model = keras.models.load_model('C:\\Users\\iniyo\\Desktop\\CNN_model\\CNN_model.h5')
# 디텍팅 함수
def Detect_face_img(user_face_img_list):
    # 배열 넣어야 되는지 아닌지 확인하기
    # 디텍팅 Haar 분류기 가져오기
    face_cascade = cv2.CascadeClassifier('C:\\haar\\haarcascade_frontalface_default.xml')
    # color ycrcb
    lower = np.array([0,133,77], dtype = np.uint8) # Ycrcb lower
    upper = np.array([255,173,127], dtype = np.uint8) # Ycrcb upper
    try:
        for file in range(len(user_face_img_list)):
            face_img = cv2.imread(user_face_dir+'{}.jpg'.format(file+1))
            gray_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_face_img, 1.2, 3)
            face_img_Ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb) 
            t, otsu_mask= cv2.threshold(gray_face_img, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            mask1 = cv2.inRange(face_img_Ycrcb, lower, upper)
            otsu_mask = cv2.bitwise_and(otsu_mask, otsu_mask, mask = mask1)
            for (x,y,w,h) in faces:
                face_crop = face_img[y:y+h, x:x+w] 
                maskin_img = otsu_mask[y:y+h, x:x+w] 
            skin = cv2.bitwise_and(face_crop, face_crop, mask = maskin_img) 
            resize_skin_img = cv2.resize(skin,(128,128))
            cvt_RGB = cv2.cvtColor(resize_skin_img, cv2.COLOR_BGR2RGB)
            detect_skin = np.asarray(cvt_RGB)
            de_sk.append(detect_skin)
    except:
        pass

Detect_face_img(user_face_dir)

de_sk = np.array(de_sk)
de_sk = de_sk / 255

pre = model.predict(de_sk)
# 시각화
for i, vl in enumerate(pre):
    if vl > 0.5:
        plt.figure()
        plt.imshow(de_sk[i,:,:])
        plt.title('warm')
        plt.axis('off')
        plt.show()
        if i == 10: break
    else :
        plt.figure()
        plt.imshow(de_sk[i,:,:])
        plt.title('cool')
        plt.axis('off')
        plt.show()
        if i == 10: break