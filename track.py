import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from helper import *
from subsample_pipeline import * 
import time
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
import collections
from Cachedata import Cachedata


ystart = 400
ystop = 656
scale = 1.5

dist_pickle = pickle.load( open("dist.pkl", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pixel_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

history = Cachedata()

def process(img):
	out_img, heatmap = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
	history.recent_heatmap.append(heatmap)
	heatmap_mean = np.mean(history.recent_heatmap, axis = 0)
	heatmap_thresh = apply_threshold(heatmap_mean)
	labels = label(heatmap_thresh)
	history.recent_labels.append(labels)
	# labels_mean = np.mean(history.recent_labels,axis=0).astype(int)
	draw_img = draw_labeled_bboxes(np.copy(img),labels)
	output_main = cv2.resize(draw_img, (1280, 540), interpolation=cv2.INTER_AREA)
	output1 = cv2.resize(np.dstack((heatmap,heatmap,heatmap)) * 255, (320, 180), interpolation=cv2.INTER_AREA)
	output2 = cv2.resize(np.dstack((heatmap_mean,heatmap_mean,heatmap_mean)) * 255, (320, 180), interpolation=cv2.INTER_AREA)
	output3 = cv2.resize(np.dstack((heatmap_thresh, heatmap_thresh, heatmap_thresh))
	                     * 255, (320, 180), interpolation=cv2.INTER_AREA)
	output4 = cv2.resize(out_img, (320, 180), interpolation=cv2.INTER_AREA)
	cv2.putText(output1, 'Heatmap', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 255, 12), 2)
	cv2.putText(output2, 'Heatmap-avg', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 255, 12), 2)
	cv2.putText(output3, 'Heatmap-thresh', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 255, 12), 2)
	cv2.putText(output4, 'multi-window with outlier', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 255, 12), 2)

   
	vis = np.zeros((720, 1280, 3))
	vis[:180, :320, :] = output1
	vis[:180, 320:640, :] = output2
	vis[:180, 640:960, :] = output3
	vis[:180, 960:, :] = output4
	vis[180:, :, :] = output_main
	
	return vis


Input_video = './project_video.mp4'
Out_video = './tracked_video.mp4'
print(Input_video)
video_clip = VideoFileClip(Input_video)
t1= time.time()
print('processing the image')
video_tracked = video_clip.fl_image(process)
print('processing is completed, processing time {}'.format(time.time() - t1))

print('writing to file...')
video_tracked.write_videofile(Out_video, audio=False)


