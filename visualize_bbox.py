from ast import arg
from email import parser
import json
import os
import argparse
from attr import attributes
import matplotlib.pyplot as plt
import cv2
import numpy as np
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--rootpath', default='/home/hdm/dataset/hope-dataset/hope_video_final/', metavar='PATH')
# parser.add_argument('--rootpath', default='/home/hdm/dataset/hope-dataset/playground_train/', metavar='PATH')

args = parser.parse_args()
# output_path = os.path.join(args.rootpath + f'../playground_{args.split}/')
# if args.rgbpath is None:
#     args.rgbpath = args.annotspath.replace('.json', '_rgb.jpg')
#     print(args.rgbpath)
#     if not os.path.exists(args.rgbpath):
#         raise FileNotFoundError(f'unable to find rgb image path {args.rgbpath}')

# mesh loading function

objects_list = json.load(open(os.path.join(args.rootpath, "data", "objects.json")))
file_list = os.listdir(args.rootpath +'image/')
image_list = [file for file in file_list if file.endswith('.jpg')]
# image_list.sort()
print(image_list)
for i in range(len(image_list)):
    img_id = i+1
    img = f'{img_id}.jpg'
    img_path = os.path.join(args.rootpath, 'image', img)
    im = cv2.imread(img_path)
    bbox_list = objects_list[i]['objects']
    for bbox in bbox_list:
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        # print(x,y,w,h)
        x1, y1 = int(x-w/2), int(y-h/2)
        # print(x1, y1) 
        x2,y2 = int(x1+w), int(y1+h)
        # print(x2, y2) 
        # print(bbox['names'])
        # im = cv2.rectangle(im, (x1, y1), (x2, y2), (0,0,255), 3)
        print(x,y,w,h)
        im = cv2.rectangle(im, (x, y), (x+w, y+h), (0,0,255), 3)
        # if (i == 1):
    print(img)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.show()
        
    