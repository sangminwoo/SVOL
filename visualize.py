import os
import cv2
import json
import numpy as np
from PIL import Image


def visualize(result_dir, video_dir):
	with open(result_dir, 'r') as json_file:
		json_list = list(json_file)

	for json_str in json_list:
		result = json.loads(json_str)
		video = result['video']
		sketch = result['sketch']
		frame = result['frame']
		gt_boxes = result['gt_boxes']
		gt_boxes = [gt_box['bbox'] for gt_box in gt_boxes]
		pred_boxes = result['pred_boxes']
		train_path = os.path.join(video_dir, 'train', video, '{:06d}'.format(frame)+'.JPEG')
		val_path = os.path.join(video_dir, 'val', video, '{:06d}'.format(frame)+'.JPEG')
		if os.path.exists(train_path):
			img_path = train_path 
		elif os.path.exists(val_path):
			img_path = val_path
		else:
			raise ValueError
		img = Image.open(img_path).convert("RGB")
		w, h = img.size
		img = np.array(img)


		# cv2.rectangle(img, top-left, bottom-right, color, line_width)
		for gt_box in gt_boxes:
			minx, miny, maxx, maxy = gt_box
			minx = round(minx * w)
			miny = round(miny * h)
			maxx = round(maxx * w)
			maxy = round(maxy * h)
			cv2.rectangle(img, (minx, miny), (maxx, maxy), (0,255,0), 1)

		for pred_box in pred_boxes:
			minx, miny, maxx, maxy, score = pred_box
			if score < 0.5:
				continue
			minx = round(minx * w)
			miny = round(miny * h)
			maxx = round(maxx * w)
			maxy = round(maxy * h)
			cv2.rectangle(img, (minx, miny), (maxx, maxy), (0,0,255), 1)

		os.makedirs(os.path.join('visualize', video, sketch), exist_ok=True)
		cv2.imwrite(os.path.join('visualize', video, sketch, f'{frame}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
	save_dir = '/mnt/server12_hard3/sangmin/code/svol/results/'
	file_name = '2022-03-04 11:43:48 PM_imagenet_vid_quickdraw_svoltr_vit_3e_3d_32f_320q_5_1_2_val.jsonl'
	result_dir = os.path.join(save_dir, file_name)

	video_dir = '/home/sangmin/drive3/sangmin/data/svol/imagenet_vid/Data/VID/'

	visualize(result_dir, video_dir)