import os
import json
from glob import glob
from xml.etree.ElementTree import parse


CLASS_HISTOGRAM_ID_SPECIFIC = {}
CLASS_HISTOGRAM_ID_AGNOSTIC = {}
CLASS_TO_VID = {}
ID_TO_NAME = {
	'n02691156': 'airplane',
	# 'n02419796': 'antelope',
	'n02131653': 'bear',
	'n02834778': 'bicycle',
	'n01503061': 'bird',
	'n02924116': 'bus',
	'n02958343': 'car',
	'n02402425': 'cow', # cattle -> cow
	'n02084071': 'dog',
	'n02121808': 'cat', # domestic cat -> cat
	'n02503517': 'elephant',
	# 'n02118333': 'fox',
	'n02510455': 'panda', # giant panda -> panda
	# 'n02342885': 'hamster',
	'n02374451': 'horse',
	'n02129165': 'lion',
	'n01674464': 'lizard',
	'n02484322': 'monkey',
	'n03790512': 'motorcycle',
	'n02324045': 'rabbit',
	# 'n02509815': 'red panda',
	'n02411705': 'sheep',
	'n01726692': 'snake',
	'n02355227': 'squirrel',
	'n02129604': 'tiger',
	'n04468005': 'train',
	'n01662784': 'turtle',
	# 'n04530566': 'watercraft',
	'n02062744': 'whale',
	'n02391049': 'zebra'
}

for name in ID_TO_NAME.values():
	CLASS_HISTOGRAM_ID_SPECIFIC[name] = 0
	CLASS_HISTOGRAM_ID_AGNOSTIC[name] = 0
	CLASS_TO_VID[name] = []


if __name__ == '__main__':
	root_dir = '/home/sangmin/drive3/sangmin/data/svol/imagenet_vid/Annotations/VID/'
	# phase = 'train'
	phase = 'val'
	anno_dirs = glob(os.path.join(root_dir, phase, '*'))
	
	for anno_dir in anno_dirs:
		folder = anno_dir.split('/')[-1]
		annos_per_video = glob(os.path.join(anno_dir, '*'))

		trackid_specific = set()
		trackid_agnostic = set()
		for anno_per_frame in annos_per_video:
			tree = parse(anno_per_frame)
			root = tree.getroot()

			for i in root.findall('object'):
				class_id = i.find('name').text
				if class_id not in ID_TO_NAME:
					continue

				trackid = int(i.find('trackid').text)
				class_label = ID_TO_NAME[class_id]

				trackid_specific.add((trackid, class_label))
				trackid_agnostic.add(class_label)

		for lbl in trackid_agnostic:
			CLASS_TO_VID[lbl].append(folder)

		for (tid, lbl) in trackid_specific:
			CLASS_HISTOGRAM_ID_SPECIFIC[lbl] += 1

		for lbl in trackid_agnostic:
			CLASS_HISTOGRAM_ID_AGNOSTIC[lbl] += 1

	print('CLASS_HISTOGRAM_ID_SPECIFIC', CLASS_HISTOGRAM_ID_SPECIFIC)
	print('CLASS_HISTOGRAM_ID_AGNOSTIC', CLASS_HISTOGRAM_ID_AGNOSTIC)
	with open('class_to_vid_' + phase + '.json', 'w') as f:
		json.dump(CLASS_TO_VID, f)

# CLASS_HISTOGRAM_ID_SPECIFIC (TRAIN)
# {'airplane': 422,'bear': 166, 'bicycle': 403, 'bus': 140,
#  'car': 1246, 'cow': 244, 'dog': 556, 'cat': 194,
#  'elephant': 250, 'panda': 168, 'horse': 213, 'lion': 134,
#  'lizard': 102, 'monkey': 272, 'motorcycle': 278, 'rabbit': 112,
#  'sheep': 179, 'snake': 96, 'squirrel': 152, 'tiger': 67,
#  'train': 176, 'turtle': 135, 'zebra': 355}

# CLASS_HISTOGRAM_ID_AGNOSTIC (TRAIN)
# {'airplane': 172, 'bear': 114, 'bicycle': 266, 'bus': 70,
# 'car': 451, 'cow': 73, 'dog': 458, 'cat': 183,
# 'elephant': 79, 'panda': 116, 'horse': 139, 'lion': 64,
# 'lizard': 92, 'monkey': 129, 'motorcycle': 149, 'rabbit': 75,
# 'sheep': 76, 'snake': 96, 'squirrel': 132, 'tiger': 56,
# 'train': 158, 'turtle': 118, 'zebra': 100}

# CLASS_HISTOGRAM_ID_SPECIFIC (VAL)
# {'airplane': 184, 'bear': 23, 'bicycle': 88, 'bus': 25,
# 'car': 229, 'cow': 52, 'dog': 89, 'cat': 34,
# 'elephant': 40, 'panda': 21, 'horse': 20, 'lion': 9,
# 'lizard': 12, 'monkey': 72, 'motorcycle': 31, 'rabbit': 13,
# 'sheep': 9, 'snake': 12, 'squirrel': 29, 'tiger': 11,
# 'train': 27, 'turtle': 17, 'zebra': 29}

# CLASS_HISTOGRAM_ID_AGNOSTIC (VAL)
# {'airplane': 55, 'bear': 20, 'bicycle': 19, 'bus': 17,
# 'car': 49, 'cow': 23, 'dog': 64, 'cat': 33,
# 'elephant': 20, 'panda': 14, 'horse': 12, 'lion': 5,
# 'lizard': 12, 'monkey': 19, 'motorcycle': 27, 'rabbit': 13,
# 'sheep': 4, 'snake': 12, 'squirrel': 22, 'tiger': 9,
# 'train': 23, 'turtle': 13, 'zebra': 12}