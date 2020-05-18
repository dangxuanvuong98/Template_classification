import glob
import shutil
import os
f=open("data_raw/train.txt").read().split("\n")[:-1]
for x in f:
    x=x.split(",")
    label=x[1]
    out_path=os.path.join("data/train/"+label)
    try:
        os.mkdir(out_path)
    except Exception as e:
        print(e)
    try:
        shutil.move("data_raw/"+x[0],out_path)
    except Exception as e:
        print(e)