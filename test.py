
from config import get_config
from load_data import *
from PIL import Image
import torch.nn.functional as F
import torch
import cv2
import numpy as np
import glob
from backbone.mobilev3.mobilenetV3 import mobilenetv3
import glob
from alignment.retinaface import Retinaface_Detector
import  torch.nn.functional  as F
import time
class Face_mask():
    def __init__(self):
        opt=get_config()
        self.classes=["face_mask","no_face_mask"]
        if(opt.back_bone=="mobiv3"):
            from backbone.mobilev3.mobilenetV3 import mobilenetv3
            self.model=mobilenetv3(input_size=opt.input_size,num_classes=2,small=False,get_weights=False).to(opt.device)
            checkpoint_path = os.path.join(opt.save_model, 'checkpoint{}.pth.tar'.format(opt.local_rank))
            checkpoint = torch.load(checkpoint_path, map_location=opt.device)
            print("load success pretrained!")
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()
        self.align=Retinaface_Detector()
        self.transform = get_transform(False,112)
    def predict_single(self,img):
        t1=time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transform(img)
        img=img.reshape(1,img.size(0),img.size(1),img.size(2)).cuda()
        output=F.softmax(self.model(img))
        print("time predict ",time.time()-t1)
    
        return self.classes[int(output.argmax(1))],float(torch.max(output))
if __name__ == "__main__":
    detector=Face_mask()
    # for x in glob.glob("/home/haobk/Mydata/face_mask_detection/test/test/0/j.png"):# 0  face_mask
    #     img=cv2.imread(x)
    #     cv2.imshow("image",img)
    #     print(detector.predict_single(img))
    #     cv2.waitKey(0)   
    align=Retinaface_Detector()
    for x in glob.glob("test/face_mask/*.jpg"):
        img=cv2.imread(x) 
        _,bboxs,faces = align.align_multi(img)
        for face in faces:
            face.show()
            img=np.array(face)
            img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            print(detector.predict_single(img))
            cv2.imshow("img",img)
            cv2.waitKey(0)

