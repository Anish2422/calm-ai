from turtle import width
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import mediapipe as mp
from imutils.video import VideoStream
import imutils
from django.conf import settings
from med_app.settings import BASE_DIR, STATIC_URL

mixer.init()

sound = mixer.Sound(BASE_DIR + '\\ai\static\\ai\media\warning.mp3')
end_sound = mixer.Sound(BASE_DIR + '\\ai\static\\ai\media\end.mp3')

leye = cv2.CascadeClassifier(BASE_DIR + '\\ai\static\\ai\\files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(BASE_DIR + '\\ai\static\\ai\\files\haarcascade_righteye_2splits.xml')

model = load_model(BASE_DIR + '\\ai\static\\ai\\files\cnnCat2.h5')
path = os.getcwd()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

image_path = BASE_DIR + '\\ai\static\\ai\images\zen_bg.jpg'
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

class VideoCamera(object):
    def __init__(self):
        self.vs = VideoStream(src=0).start()
        self.prev = time.time()
        self.TIMER = int(60)
        self.finish = True
        
    def __del__(self):
        cv2.destroyAllWindows()

    def get_frame(self, rpred, lpred, lbl, lbl_pred):
        frame = self.vs.read()
        frame = imutils.resize(frame, width=650)
        height , width, _ = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)

        results = selfie_segmentation.process(RGB)
        mask = results.segmentation_mask
        condition = np.stack(
        (mask,) * 3, axis=-1) > 0.6

        bg_image = cv2.imread(image_path)
        bg_image = cv2.resize(bg_image, (width, height))

        cv2.rectangle(frame, (150,height-50) , (400,height) , (0,0,0) , thickness=cv2.FILLED )

        for (x,y,w,h) in right_eye:
            r_eye=frame[y:y+h,x:x+w]
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye=  r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = model.predict(r_eye)
            rpred=np.argmax(rpred,axis=1)
            if(rpred[0]==1):
                lbl_pred=lbl[1]    #'Distracted' 
            if(rpred[0]==0):
                lbl_pred=lbl[0]    #'Meditating'
            break

        for (x,y,w,h) in left_eye:
            l_eye=frame[y:y+h,x:x+w]
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = model.predict(l_eye)
            lpred = np.argmax(lpred,axis=1)
            if(lpred[0]==1):
                lbl_pred=lbl[1]    #'Distracted'   
            if(lpred[0]==0):
                lbl_pred=lbl[0]    #'Meditating'
            break

        cur = time.time() #to check if the score has increased after the eye was open

        prev_time = self.TIMER

        if(rpred[0]==1 and lpred[0]==1):
            if self.TIMER>=0:
                if cur-self.prev >= 1:
                    self.prev = cur
                    self.TIMER = self.TIMER+1
            cv2.rectangle(frame, (0,height-50) , (150,height) , (0,0,255) , thickness=cv2.FILLED )
            cv2.putText(frame,lbl_pred,(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

        else:
            
            if self.TIMER>=0:
                if cur-self.prev >= 1:
                    self.prev = cur
                    self.TIMER = self.TIMER-1
            cv2.rectangle(frame, (0,height-50) , (150,height) , (0,255,0) , thickness=cv2.FILLED )
            cv2.putText(frame,lbl_pred,(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

            # combine frame and background image using the condition
            frame = np.where(condition, frame, bg_image)

            # if self.TIMER == int_time/2:     
            #     cv2.imwrite(os.path.join(path,'zen.jpg'),frame)
        
        cv2.putText(frame,'Meditation Meter:'+str(self.TIMER),(150,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)  

        if(self.TIMER==1):
            frame=np.zeros([512,512,3], np.uint8)

        if(self.TIMER<1):
            self.TIMER = 0   
            self.finish = False
            try:
                end_sound.play()
            except:  # isplaying = False
                pass

        if(self.TIMER>prev_time and self.TIMER%4==0):
            #person is distracted, so we set off the warning message
            try:
                sound.play()
                
            except:  # isplaying = False
                pass
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), self.finish

