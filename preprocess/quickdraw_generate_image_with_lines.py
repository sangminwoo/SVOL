import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm, trange


root_dir = '/mnt/server15_hard2/sangmin/data/quickdraw/full/raw/'
save_dir = '/mnt/server15_hard2/sangmin/data/quickdraw/sketch/'
# classes = ['airplane', 'bear', 'bicycle', 'bird', 'bus',
# 		   'car', 'cat', 'cow', 'dog', 'elephant', 'horse',
# 		   'lion', 'monkey', 'motorbike', 'panda',
# 		   'rabbit', 'sheep', 'snake', 'squirrel',
# 		   'tiger', 'train', 'sea turtle', 'whale', 'zebra']
classes = ['motorbike', 'sea turtle']
		   # motorbike -> motorcycle
		   # sea turtle -> turtle
size = (224, 224)
dpi = 96

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
		
		plt.figure(figsize=(size[0]/dpi, size[1]/dpi), dpi=dpi)
		for draw in raw['drawing'][i]:
		    drawnp = np.array(draw)
		    drawnp = drawnp.transpose()[:, :2]
		    for j in range(len(drawnp)):
		        plt.plot(drawnp[:, 0][j:j+2], drawnp[:, 1][j:j+2], 'k-')
		plt.axis('off')
		plt.gca().invert_yaxis()

		os.makedirs(save_dir + sub_dir[:-7], exist_ok=True)
		plt.savefig(os.path.join(save_dir, sub_dir[:-7], sub_dir[:-7] + f'{count:04}' + '.png'), dpi=dpi)
		plt.close()
		
		count += 1

		if count > 1000:
			break