import cv2
import numpy as np

import time
import sys
import os
import cProfile
CONFIDENCE = 0.9
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
import os

print(os.path.realpath(__file__))
# the neural network configuration
config_path = "/Users/simonh/Downloads/lastproj/cfg/yolov3.cfg"
# the YOLO net weights file
weights_path = "/Users/simonh/Downloads/lastproj/weights/yolov3.weights"
# weights_path = "weights/yolov3-tiny.weights"
NMSTHRESHOLD = 0.3
# loading all the class labels (objects)
labels = open("/Users/simonh/Downloads/lastproj/data/coco.names").read().strip().split("\n")
# generating colors for each object for later plotting
if not os.path.isfile(weights_path):
    print(f"Weights file not found at {weights_path}")
else:
    print(f"Weights file found at {weights_path}")
if not os.path.isfile(config_path):
    print(f"Config file not found at {config_path}")
else:
    print(f"Config file found at {config_path}")

colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
np.random.seed(10)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")


net = cv2.dnn.readNetFromDarknet(config_path, weights_path)



cap = cv2.VideoCapture('fortnite.mp4')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameCount = 1200
buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

fc = 0
ret = True
print(frameCount)
while (fc < frameCount  and ret):
    ret, buf[fc] = cap.read()
    fc += 1

cap.release()



image = cv2.imread('/Users/simonh/Downloads/lastproj/images/streets.jpg')

file_name = os.path.basename('/Users/simonh/Downloads/lastproj/images/streets.jpg')
filename, ext = file_name.split(".")



import timeit

h, w , p = image.shape
# create 4D blob
def getfr(image, w, h, tryint):
    
    if tryint == 1:
        profiler = cProfile.Profile()
        profiler.enable()
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    time.sleep(2)
    net.setInput(blob)
    # get all the layer names
    ln = net.getLayerNames()
    try:
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        # in case getUnconnectedOutLayers() returns 1D array when CUDA isn't available
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    # feed forward (inference) and get the network output
    # measure how much it took in seconds
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start



    font_scale = 1
    thickness = 1
    boxes, confidences, class_ids = [], [], []
    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # discard out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)




    #people = np.array([])

    for i in range(len(boxes)):
        # extract the bounding box coordinates
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
        text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
        # calculate text width & height to draw the transparent boxes as background of the text
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        text_offset_x = x
        text_offset_y = y - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
        # add opacity (transparency to the box)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        # now put the text (label: confidence %)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
    #print(people)


    if(tryint == 1):
        profiler.disable()
        profiler.dump_stats('getfr_output.prof')

    
    #unique_objects, counts = np.unique(people, return_counts=True)

    # Print the results
    #for obj, count in zip(unique_objects, counts):
    #    print(f"{obj}: {count}")
    return image
it=0
tryint = 0
duration = frameCount
fps = 30
out = cv2.VideoWriter('/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frameWidth, frameHeight), True)
for i in range(frameCount):
    
    image = (getfr(buf[i], frameWidth, frameHeight, i))
    
    print(f"Frame {i}/{frameCount} Processed")
    print(i)
    out.write(image)
    
out.release()
#video_name = "vid"
#image_folder = "C:/Users/digipen/Downloads/lastproj/images"

    