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
import open3d as o3d


parser = argparse.ArgumentParser()
parser.add_argument('annotspath', metavar='PATH')
parser.add_argument('--rgbpath', default=None, metavar='PATH',
                    help='Path to RGB image\n(optional, default: annotspath.replace(".json","_rgb.jpg"))')
parser.add_argument('--meshdir', default='meshes/eval/', metavar='PATH',
                    help='Path to object meshes\n(optional, default: meshes/eval/)')

args = parser.parse_args()

if not os.path.exists(args.annotspath):
    raise FileNotFoundError(f'unable to find annotations path {args.annotspath}')

if args.rgbpath is None:
    args.rgbpath = args.annotspath.replace('.json', '_rgb.jpg')
    print(args.rgbpath)
    if not os.path.exists(args.rgbpath):
        raise FileNotFoundError(f'unable to find rgb image path {args.rgbpath}')

# mesh loading function
def load_mesh(object_class):
    mesh_fn = os.path.join(args.meshdir, f'{object_class}.obj')
    if not os.path.exists(mesh_fn):
        raise FileNotFoundError(f'Unable to open mesh path {mesh_fn}')
    return trimesh.load(mesh_fn)

# load scene annotations
print(f'Loading annotations ({args.annotspath})...')
annots = json.load(open(args.annotspath))

# get camera intrinsics
camera_intrinsics = np.array(annots['camera']['intrinsics'])
w, h = annots['camera']['width'], annots['camera']['height']
fx, fy, _ = np.diag(camera_intrinsics)
cx, cy, _ = camera_intrinsics[:, -1]

# load objects
print(f'Loading object meshes (from {args.meshdir})...')
mesh_names = []
cnt = 0
im = cv2.imread(args.rgbpath)
for obj in annots['objects']:
    # if (cnt != 10):
    #     cnt += 1
    #     continue
    mesh = load_mesh(obj['class'])
    # print(mesh.bounds)
    # corners = trimesh.bounds.corners(mesh.bounding_box.bounds)
    # print(corners)
    # corners = trimesh.bounds.corners(mesh.bounds)
    # print(corners)
    transformation_matrix = np.array(obj['pose'])
    mesh.apply_transform(transformation_matrix)
    # mesh_name = scene.add_geometry(mesh)
    # mesh_names.append(mesh_name)
    corners = trimesh.bounds.corners(mesh.bounding_box.bounds)
    max_x, max_y = 0,0
    min_x, min_y = w,h
    for corner in corners:
        print(camera_intrinsics)
        print(np.transpose(corner))
        new = np.array([corner])
        transformed_corner = np.matmul(camera_intrinsics, new.T)
        print(transformation_matrix)
        transformed_corner /= transformed_corner[2]
        print(transformed_corner)
        # x = int(w/2+(corner[0]*corner[2]))
        # y = int(h/2+(corner[1]*corner[2]))
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
        print(x, y)
        # im = cv2.circle(im, (int(w/2+corner[0]), int(h/2+corner[1])), 5, (0,0,255),-1)
        im = cv2.circle(im, (x,y), 5, (0,0,255),-1)
    im = cv2.rectangle(im, (min_x,min_y),(max_x,max_y), (0,0,255), 3)
    cnt += 1
    
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.show()
# im = cv2.circle(im, (50,1050), 10, (0,0,0),-1)
