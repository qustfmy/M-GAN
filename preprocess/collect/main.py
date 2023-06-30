import cv2, threading  
import numpy as np
from inference import AnalysisModel
from preprocess import pre_det, pre_val
from postprocess import post_det

from tkinter import Tk, Button

def change_face():
    global change_flag
    change_flag = True

def main():
    global change_flag
    cap = cv2.VideoCapture(0)
    model_det = AnalysisModel('inference/face_detection/__model__', 
                            'inference/face_detection/__params__',
                            True,
                            False)

    model_val = AnalysisModel('inference/face_verification/__model__', 
                            'inference/face_verification/__params__',
                            False,
                            True)
    tmp = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        sucess, img = cap.read()
        img_det = pre_det(img, 0.3)
        result_det = model_det.predict_det(img_det)
        img, crops, bboxes = post_det(img, result_det)       
        if type(tmp) is np.ndarray:
            for crop, bbox in zip(crops, bboxes):
                img_val = pre_val(tmp, crop)
                x1, y1 = bbox[:2]
                result_val = model_val.predict_val(img_val)
                if np.argmax(result_val[0]):
                    img = cv2.putText(img, 'Success', (x1, y1-4), font, 0.6, (0, 255, 0), 2)                
                else:
                    img = cv2.putText(img, 'Faild', (x1, y1-4), font, 0.6, (0, 0, 255), 2)          
        if (len(crops)>0)  and change_flag:
            tmp = crops[0]
            crop = crops[0]
            cv2.imshow('Face', crop)
            change_flag=False 
        cv2.imshow('Face recognition', img)
        k = cv2.waitKey(1)
        if k == 27:
            #通过esc键退出摄像
            cv2.destroyAllWindows()
            break
            
if __name__=='__main__':
    global change_flag
    change_flag = False
    root = Tk()
    root.title('Button')
    button = Button(root, text ="点击抓取人脸图片", command = change_face)
    button.pack()
    main_thread = threading.Thread(target=main)
    main_thread.start()
    root.mainloop()