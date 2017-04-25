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

	
	# heatmap_mean = np.mean(history.recent_heatmap).ravel()
	heatmap_thresh = apply_threshold(heatmap)
	labels = label(heatmap_thresh)
	draw_img = draw_labeled_bboxes(np.copy(img),labels)


	output_main = cv2.resize(draw_img, (1280, 540), interpolation=cv2.INTER_AREA)
	return output_main

    # output1 = cv2.resize(heatmap, (320, 180), interpolation=cv2.INTER_AREA)
    # output2 = cv2.resize(heatmap_mean, (320, 180), interpolation=cv2.INTER_AREA)
    # output3 = cv2.resize(np.dstack((combined, combined, combined))
                         # * 255, (320, 180), interpolation=cv2.INTER_AREA)
    # output4 = cv2.resize(vid2, (320, 180), interpolation=cv2.INTER_AREA)
    # vis = np.zeros((720, 1280, 3))
    # vis[:180, :320, :] = output1
    # vis[:180, 320:640, :] = output1
    # vis[:180, 640:960, :] = output1
    # vis[:180, 960:, :] = output1
    # vis[180:, :, :] = output_main



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


