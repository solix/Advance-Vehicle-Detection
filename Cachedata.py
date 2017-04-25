import collections


# Define a class to receive the characteristics of each line detection
class Cachedata():

    def __init__(self):
        #  detected in the last iteration?
        self.falsePositive = False
        # values of the last n fits of the detections
        self.recent_heatmap = collections.deque([], maxlen=10)