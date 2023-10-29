import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm, trange


pad_size = 5
root_dir = '/mnt/server15_hard2/sangmin/data/quickdraw/full/raw/'
save_dir = '/mnt/server15_hard2/sangmin/data/quickdraw/sketch/'
classes = ['airplane', 'bear', 'bicycle', 'bird', 'bus',
		   'car', 'cat', 'cow', 'dog', 'elephant', 'horse',
		   'lion', 'monkey', 'motorcycle', 'panda',
		   'rabbit', 'sheep', 'snake', 'squirrel',
		   'tiger', 'train', 'turtle', 'whale', 'zebra']

for sub_dir in tqdm(os.listdir(root_dir),
					desc='Extracting Images',
					total=len(os.listdir(root_dir))):
	if sub_dir[:-7] not in classes:
		continue
	raw = pd.read_json(root_dir + sub_dir, lines=True)
	count = 1
	for i in trange(len(raw), desc=sub_dir[:-7]):
		if not raw['recognized'][i]:
			continue
		xcoords = []
		ycoords = []
		for (x, y, t) in raw['drawing'][i]:
		    xcoords.extend(x)
		    ycoords.extend(y)
		coords = np.array(list(set(zip(xcoords, ycoords))), dtype=int)

		min_r = min(coords[:, 0])
		min_c = min(coords[:, 1])

		rows = max(coords[:, 0]) - min(coords[:, 0])
		cols = max(coords[:, 1]) - min(coords[:, 1])

		canvas = np.ones((rows+pad_size, cols+pad_size), dtype=np.uint8)*255

		for coord in coords:
		    for pad_r in range(pad_size):
		        for pad_c in range(pad_size):
		            canvas[coord[0]-min_r-1+pad_r, coord[1]-min_c-1+pad_c] = 0 # origin
		            canvas[coord[0]-min_r-1+pad_r, coord[1]-min_c-1-pad_c] = 0 # origin
		            canvas[coord[0]-min_r-1-pad_r, coord[1]-min_c-1+pad_c] = 0 # origin
		            canvas[coord[0]-min_r-1-pad_r, coord[1]-min_c-1-pad_c] = 0 # origin
	    
		canvas = canvas.transpose()
		img = Image.fromarray(canvas).resize((224, 224), resample=Image.Resampling.LANCZOS)
		
		os.makedirs(save_dir + sub_dir[:-7], exist_ok=True)

		img.save(os.path.join(save_dir, sub_dir[:-7], sub_dir[:-7] + f'{count:04}' + '.png'))

		count += 1

		if count > 1000:
			break