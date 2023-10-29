import os
import json
from xml.etree.ElementTree import parse

root_dir = '/home/sangmin/drive2/data/svol/imagenet_vid/Annotations/VID/'
# phase = 'train/'
# phase = 'val/'
phase = 'all/'

ID_TO_CLASS = {'n02691156': 'airplane',
			   'n02419796': 'antelope',
			   'n02131653': 'bear',
			   'n02834778': 'bicycle',
			   'n01503061': 'bird',
			   'n02924116': 'bus',
			   'n02958343': 'car',
			   'n02402425': 'cow', # cattle -> cow
			   'n02084071': 'dog',
			   'n02121808': 'cat', # domestic cat -> cat
			   'n02503517': 'elephant',
			   'n02118333': 'fox',
			   'n02510455': 'panda', # giant panda -> panda
			   'n02342885': 'hamster',
			   'n02374451': 'horse',
			   'n02129165': 'lion',
			   'n01674464': 'lizard',
			   'n02484322': 'monkey',
			   'n03790512': 'motorcycle',
			   'n02324045': 'rabbit',
			   'n02509815': 'red panda',
			   'n02411705': 'sheep',
			   'n01726692': 'snake',
			   'n02355227': 'squirrel',
			   'n02129604': 'tiger',
			   'n04468005': 'train',
			   'n01662784': 'turtle',
			   'n04530566': 'watercraft',
			   'n02062744': 'whale',
			   'n02391049': 'zebra'}

annos = {}
for sub_dir in sorted(os.listdir(root_dir + phase)):  # ILSVRC2015_train_00000000
	num_frames = 0
	objects = set()
	for idx, subsub_dir in enumerate(sorted(os.listdir(root_dir + phase + sub_dir))):  # 000000.xml
		tree = parse(os.path.join(root_dir + phase + sub_dir, subsub_dir))
		root = tree.getroot()

		if idx == 0:
			width = int(root.find('size').find('width').text)
			height = int(root.find('size').find('height').text)
			annos[sub_dir] = {
				'size': [width, height],
				'num_frames': None,
				'objects': None,
				'frames': {}
			}

		obj_annos = []
		for obj in root.findall('object'):
			track_id = int(obj.find('trackid').text)
			label = ID_TO_CLASS[obj.find('name').text]
			objects.add(label)

			bbox = [int(obj.find('bndbox').find('xmin').text),
					int(obj.find('bndbox').find('ymin').text),
					int(obj.find('bndbox').find('xmax').text),
					int(obj.find('bndbox').find('ymax').text)]
			obj_annos.append({
				'track_id': track_id,
				'label': label,
				'bbox': bbox
			})
		annos[sub_dir]['frames'][subsub_dir[:-4]] = obj_annos
		num_frames += 1

	annos[sub_dir]['num_frames'] = num_frames
	annos[sub_dir]['objects'] = list(objects)

# with open(root_dir + 'train.json', 'w') as f:
# with open(root_dir + 'val.json', 'w') as f:
with open(root_dir + 'all.json', 'w') as f:
	json.dump(annos, f)