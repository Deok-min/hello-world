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
import pyglet
import PIL
label = ["AlphabetSoup",
        "BBQSauce",
        "Butter",
        "Cherries",
        "ChocolatePudding",
        "Cookies",
        "Corn",
        "CreamCheese",
        "GranolaBars",
        "GreenBeans",
        "Ketchup",
        "MacaroniAndCheese",
        "Mayo",
        "Milk",
        "Mushrooms",
        "Mustard",
        "OrangeJuice",
        "Parmesan",
        "Peaches",
        "PeasAndCarrots",
        "Pineapple",
        "Popcorn",
        "Raisins",
        "SaladDressing",
        "Spaghetti",
        "TomatoSauce",
        "Tuna",
        "Yogurt",
        "table"]
parser = argparse.ArgumentParser()
parser.add_argument('--rootpath', default='/home/hdm/dataset/hope-dataset/hope_video_final/', metavar='PATH')
# parser.add_argument('--rootpath', default='/home/hdm/dataset/hope-dataset/playground/', metavar='PATH')
parser.add_argument('--split', default='train', metavar='PATH')
# parser.add_argument('--jsonpath', default=None, metavar='PATH',
#                     help='Path to .json file\n(optional, default: root')
parser.add_argument('--meshdir', default='meshes/eval/', metavar='PATH',
                    help='Path to object meshes\n(optional, default: meshes/eval/)')

args = parser.parse_args()
# output_path = os.path.join(args.rootpath + f'../playground_{args.split}/')
output_path = os.path.join(args.rootpath + 'data/')
print(output_path)

if not os.path.exists(output_path):
    os.makedirs(output_path)

def load_mesh(object_class):
    mesh_fn = os.path.join(args.meshdir, f'{object_class}.obj')
    if not os.path.exists(mesh_fn):
        raise FileNotFoundError(f'Unable to open mesh path {mesh_fn}')
    return trimesh.load(mesh_fn)

# load scene annotations
hope_image_data_json = []
hope_objects_json = []
hope_relashionship_json = []
hope_attributes_json = []
image_id = 0
obj_id = 0
relation_id = 0

def inside(point, size):
    return (point[0]>0 and point[0]<size[0] and point[1]>0 and point[1]<size[1])

attributes_list = json.load(open("attributes_temp.json"))
print(f'Loading annotations ({args.rootpath})...')

image_path = os.path.join(args.rootpath, 'image')
image_list = os.listdir(image_path)
for i in range(len(image_list)):
    image_id = i+1
    img = f'{image_id}.jpg'
    annot_path = img.replace('jpg', 'json')
    img_path = os.path.join(image_path, img)
    pose_path = os.path.join(args.rootpath, 'pose', annot_path)
    relation_gt_path = os.path.join(args.rootpath, 'relations', annot_path)
    
    annots = json.load(open(pose_path))
    relation_gt = json.load(open(relation_gt_path))

        # get camera intrinsics
    camera_intrinsics = np.array(annots['camera']['intrinsics'])
    width, height = annots['camera']['width'], annots['camera']['height']
    hope_image_data_json.append({'image_id':str(image_id), 'width':width, 'height':height, 'file_name':img})
    
    # load objects
    # print(f'Loading object (from {scene}_{img})...')
    mesh_names = []
    # im = cv2.imread(img_path)
    
    objects = []
    relations = []
    attribute = []
    for obj in annots['objects']:
        mesh = load_mesh(obj['class'])
        transformation_matrix = np.array(obj['pose'])
        mesh.apply_transform(transformation_matrix)
        # mesh_name = scene.add_geometry(mesh)
        # mesh_names.append(mesh_name)
        corners = trimesh.bounds.corners(mesh.bounding_box.bounds)

        center = np.average(corners,axis=0)
        center = np.array([center])
        center = np.matmul(camera_intrinsics, center.T)
        center /= center[2]
        if (not inside(center, (width,height))):
            continue
        max_x, max_y = 0,0
        min_x, min_y = width, height
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

            if min_x < 0:
                min_x = 1
            if  min_y < 0:
                min_y = 1
            if max_x >= width:
                max_x = width-2
            if max_y >= height:
                max_y = height-2
        
        # print(min_x, min_y, max_x, max_y)
        # x, y = int((min_x+max_x)/2), int((min_y+max_y)/2)
        x, y = min_x, min_y
        w, h = max_x-min_x, max_y-min_y
        
        obj_id += 1
        objects.append({'object_id':str(label.index(obj['class'])), 'x':x, 'y':y, 'w':w, 'h':h, 'names':[obj['class']], 'synsets':[]})
        # im = cv2.rectangle(im, (x,y), (x+w, y+h), (0,0,255), 1)
    obj_id += 1
    # objects.append({'object_id':str(label.index(obj['class'])), 'x':int(width/2), 'y':int(height/2), 'w':width-1, 'h':height-1, 'names':['table'], 'synsets':[]})
    objects.append({'object_id':str(label.index(obj['class'])), 'x':1, 'y':1, 'w':width-1, 'h':height-1, 'names':['table'], 'synsets':[]})
    # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    # plt.show()

    for sub in objects:
        ## create relationship
        if sub['names'][0] in relation_gt.keys():
            relation_id += 1
            # relation = {}
            object_name = relation_gt[sub['names'][0]]['object']
            for obj in objects:
                if object_name == obj['names'][0]:
                    relations.append({'relationship_id':str(1), 'predicate':relation_gt[sub['names'][0]]['predicate'], 'synsets':[],
                        'subject':sub, 'object':obj})
        
        ## create attributes
        new_obj = sub.copy()
        for attr in attributes_list:
            if sub['names'][0] in attr.keys():
                new_obj["attributes"] = attr[sub['names'][0]]
                break
        attribute.append(new_obj)

    hope_objects_json.append({'image_id':str(image_id), 'objects':objects})
    hope_relashionship_json.append({'image_id':str(image_id), 'relationships':relations})
    hope_attributes_json.append({'image_id':str(image_id), 'attributes':attribute})
    # cv2.imwrite((output_path + f'{image_id}.jpg'), im)

print(hope_objects_json)
with open(output_path+'image_data.json', 'w') as image_data_json_file:
    json.dump(hope_image_data_json, image_data_json_file)

with open(output_path+'objects.json', 'w') as object_json_file:
    json.dump(hope_objects_json, object_json_file)

with open(output_path+'relationships.json', 'w') as relation_json_file:
    json.dump(hope_relashionship_json, relation_json_file)

with open(output_path+'attributes.json', 'w') as attribute_json_file:
    json.dump(hope_attributes_json, attribute_json_file)

