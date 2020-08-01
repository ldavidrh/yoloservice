from yoloservice.slimyolov3.slimyolo import SlimYoloObjectDetection
import numpy as np
from PIL import Image

yolo = SlimYoloObjectDetection()

img = Image.open("cat.jpg")

array = np.array(img)

yolo.non_processed_frames.append(array)

