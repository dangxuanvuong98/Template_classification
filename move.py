import shutil
import os
f=open("data/MAFA/test.txt","r").read().split("\n")[:-1]
for x in f:
    path,label=x.split(" ")
    try:
        print(path)
        shutil.move(path,"data/MAFA/"+label)
    except:
        print(path)