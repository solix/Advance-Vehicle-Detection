import collections
import numpy as np


# Define a class to receive the characteristics of each line detection
class Cachedata():

    def __init__(self):
        #  detected in the last iteration?
        self.falsePositive = False
        # values of the last n fits of the detections
        self.recent_heatmap = collections.deque([], maxlen=3)
        self.mean_heatmap = np.mean(self.recent_heatmap,axis=0)

                # values of the last n fits of the detections
        self.recent_labels = collections.deque([], maxlen=3)
        self.mean_labels = np.mean(self.recent_heatmap,axis=0)