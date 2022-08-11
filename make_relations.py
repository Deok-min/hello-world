import json
import os

cwd = os.getcwd()
# path = os.path.join(cwd,'playground')
path = os.path.join(cwd,'hope_video')
scene_list = os.listdir(path)
for i, scene in enumerate(scene_list):
    scene_path = os.path.join(path, scene)
    annotpath = os.path.join(scene_path, '0000.json')
    annots = json.load(open(annotpath))

    relations = dict()
    for obj in annots['objects']:
        # rel = {'predicate':'on', 'subject':obj['class'],'object':'table'}
        relations[obj['class']] = {'predicate':'on', 'object':'table'}
    if i==2:
        relations['PeasAndCarrots'] = {'predicate':'on', 'object':'Cookies'}
    
    output = os.path.join(scene_path, 'relation.json')
    with open(output, 'w') as relation_file:
        json.dump(relations, relation_file)
