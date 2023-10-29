import os
import random
import numpy as np
from glob import glob
from PIL import Image


def array_to_pil(root_dir, sample_size):
	data_dir = os.path.join(root_dir, 'full', 'numpy_bitmap')
	save_dir = os.path.join(root_dir, 'images')
	os.makedirs(save_dir, exist_ok=True)

	QuickDraw_AND_ImageNet = [
		'airplane', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cat', 'cow', 'dog',
		'elephant', 'horse', 'lion', 'monkey', 'motorbike', 'panda', 'rabbit',
		'sheep', 'snake', 'squirrel', 'tiger', 'train', 'sea turtle', 'whale', 'zebra'
	]

	for category in QuickDraw_AND_ImageNet:
		save_category_as = category
		if category == 'motorbike': # motorbike -> motorcycle
			save_category_as = 'motorcycle'
		if category == 'sea turtle': # sea turtle -> turtle
			save_category_as = 'turtle'
		category_dir = os.path.join(save_dir, save_category_as)
		os.makedirs(category_dir, exist_ok=True)

		array_dir = os.path.join(data_dir, category+'.npy')
		category_array = np.load(array_dir)
		length = category_array.shape[0]
		sampled_idxs = random.sample(range(length), sample_size)

		for idx, sample in enumerate(category_array[sampled_idxs]):
			# convert black & white
			# resize to 224x224
			img = Image.fromarray(255-sample.reshape(28, 28)).resize((224, 224), Image.BICUBIC)
			img.save(os.path.join(category_dir, f'{save_category_as}_{idx}.png'))

if __name__=='__main__':
	root_dir = '/home/sangmin/drive3/sangmin/data/quickdraw/'
	sample_size = 1000
	array_to_pil(root_dir, sample_size)