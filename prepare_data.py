from alignment.retinaface import Retinaface_Detector
import cv2
import os
import pickle
import numpy as np
from tqdm import tqdm

import glob
from PIL import Image
align=Retinaface_Detector()
label=[0,1,2,3,4,5,6]
path="data/test"
for lb in label:
    path_lb=os.path.join(path,str(lb),"*.jpg")
    for x in tqdm(glob.glob(path_lb)):
        img=cv2.imread(x)
        os.remove(x)
        _,bboxs,faces = align.align_multi(img)
        if(len(faces)==1):
            numpy_image=np.array(faces[0])  
            opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            img=cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(x,img)
        
