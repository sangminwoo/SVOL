import os
import json
from glob import glob


if __name__ == '__main__':
	root_dir = '/home/sangmin/drive3/sangmin/data/svol/'
	# dataset = 'sketchy'
	# dataset = 'tu_berlin'
	dataset = 'quickdraw'
	mode = 'freeze'
	norm = 'after_norm'
	feature = 'class_token'
	categories = sorted(os.listdir(os.path.join(root_dir, dataset+'_features', mode, norm, feature)))

	CLASS_TO_SKETCH_TRAIN = {}
	CLASS_TO_SKETCH_VAL = {}

	for category in categories:
		sketches = os.listdir(os.path.join(root_dir, dataset+'_features', mode, norm, feature, category))
		sketches = [sketch.split('.')[0] for sketch in sketches]
		sketches_train = sketches[:int(len(sketches)*0.8)]
		sketches_val = sketches[int(len(sketches)*0.8):]
		CLASS_TO_SKETCH_TRAIN[category] = sketches_train
		CLASS_TO_SKETCH_VAL[category] = sketches_val

	with open(f'class_to_{dataset}_train.json', 'w') as f:
		json.dump(CLASS_TO_SKETCH_TRAIN, f)

	with open(f'class_to_{dataset}_val.json', 'w') as f:
		json.dump(CLASS_TO_SKETCH_VAL, f)