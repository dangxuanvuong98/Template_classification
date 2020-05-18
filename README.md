# Template_classification
Hi every body
I create two back bone : mobinev3(large or small) and efficient(b0-b7)
All parameters can edited in config.py

Train:
  Put data in folder data with format (data/0 , data/1,data/2 ...) with name folder 0,1,2.. are label
  Then edit config.py 
  And run file train.py

Open Visualize with visdom:
  pip3 install visdom
  python3 -m visdom.server parallel python3 train.py
