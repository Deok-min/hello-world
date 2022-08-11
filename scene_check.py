from ast import arg
from email import parser
import json
import os
import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
import trimesh
import pyglet
import PIL

parser = argparse.ArgumentParser()
parser.add_argument('--rootpath', default='/home/hdm/dataset/hope-dataset/hope_video/', metavar='PATH')
# parser.add_argument('--rootpath', default='/home/hdm/dataset/hope-dataset/playground/', metavar='PATH')
parser.add_argument('--split', default='train', metavar='PATH')
# parser.add_argument('--jsonpath', default=None, metavar='PATH',
#                     help='Path to .json file\n(optional, default: root')
parser.add_argument('--meshdir', default='meshes/eval/', metavar='PATH',
                    help='Path to object meshes\n(optional, default: meshes/eval/)')

args = parser.parse_args()

# mesh loading function
def load_mesh(object_class):
    mesh_fn = os.path.join(args.meshdir, f'{object_class}.obj')
    if not os.path.exists(mesh_fn):
        raise FileNotFoundError(f'Unable to open mesh path {mesh_fn}')
    return trimesh.load(mesh_fn)

# load scene annotations
hope_image_data_json = []
hope_objects_json = []
image_id = 0
obj_id = 0
total_object = 0

scene_path = args.rootpath + 'scene_0000'
file_list = os.listdir(scene_path)
image_list = [file for file in file_list if file.endswith(".jpg")]
for img in image_list:
    image_id += 1
    img_path = os.path.join(scene_path, img)
    annot_path =  img_path.replace('_rgb.jpg', '.json')
    annots = json.load(open(annot_path))

    # get camera intrinsics
    camera_intrinsics = np.array(annots['camera']['intrinsics'])
    width, height = annots['camera']['width'], annots['camera']['height']
    hope_image_data_json.append({"image_id":image_id, "url":"", "width":width, "height":height, "coco_id":image_id, "flickr_id":image_id})
    
    # load objects
    im = cv2.imread(img_path)
    
    num_objects_in_image = len(annots['objects'])
    visible_objects = 0
    if num_objects_in_image > total_object:
        total_object = num_objects_in_image
    for obj in annots['objects']:
        obj_id += 1
        mesh = load_mesh(obj['class'])
        transformation_matrix = np.array(obj['pose'])
        mesh.apply_transform(transformation_matrix)
        corners = trimesh.bounds.corners(mesh.bounding_box.bounds)
        max_x, max_y = 0,0
        min_x, min_y = width, height
        
    
        # Get min, max in 3D (before projection)
        max_3d = np.max(corners, axis=0)
        min_3d = np.min(corners, axis=0)
        print("@@")
        print(max_3d, min_3d)
        max_3dd = np.matmul(camera_intrinsics, np.array([max_3d]).T)
        max_3dd /= max_3dd[2]
        min_3dd = np.matmul(camera_intrinsics, np.array([min_3d]).T)
        min_3dd /= min_3dd[2]
            

        for corner in corners:
            new = np.array([corner])
            transformed_corner = np.matmul(camera_intrinsics, new.T)
            transformed_corner /= transformed_corner[2]
            x = int(transformed_corner[0])
            y = int(transformed_corner[1])
            if min_x > x:
                min_x = x
            elif max_x < x:
                max_x = x
            if min_y > y:
                min_y = y
            elif max_y < y:
                max_y = y
        # print(f'({min_x, min_y}),({max_x},{max_y}) // ({min_3dd[0], min_3dd[1]}),({max_3dd[0]},{max_3dd[1]})')
        # print("@")
        temp = np.average(corners,axis=0)
        temp = np.array([temp])
        transformed_world = np.matmul(camera_intrinsics, temp.T)
        transformed_world /= transformed_world[2]
        
        x, y = int((min_x+max_x)/2), int((min_y+max_y)/2)
        w, h = max_x-min_x, max_y-min_y
        im = cv2.rectangle(im, (min_x,min_y),(max_x,max_y), (0,0,255), 3)
        im = cv2.circle(im, (x,y), 3,(0,0,255), -1)
        
        im = cv2.rectangle(im, (min_3dd[0],min_3dd[1]),(max_3dd[0],max_3dd[1]), (255,0,0), 3)
        im = cv2.circle(im, (int(transformed_world[0]), int(transformed_world[1])), 3,(255,0,255), -1)
        
        if (x > 0 and y>0 and x <width and y<height):
            visible_objects += 1
        # objects.append({"object_id":obj_id, "x":x, "y":y, "w":w, "h":h, "name":obj['class'], "synsets":[]})
    # hope_objects_json.append({"image_id":image_id, "objects":objects})
    print(f'vis / existing : {visible_objects} / {num_objects_in_image}')
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.show()

# with open(output_path+'image_data.json', 'w') as image_data_json_file:
#     json.dump(hope_image_data_json, image_data_json_file)

# with open(output_path+'objects.json', 'w') as object_json_file:
#     json.dump(hope_objects_json, object_json_file)
