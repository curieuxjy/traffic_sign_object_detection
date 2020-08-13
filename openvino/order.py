# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import socket
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
# SOCKET
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
dest = ("localhost", 9999)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed
video = cv2.VideoCapture(0)
# ret = video.set(3, 720)
# ret = video.set(4,720)

a_dict = {"bicycle": 1, "child":2, "const":3, "bump":2, "cross":4, "inter":5, "parking":6, "bus":7, "left":8, "right":9}
while(True):
    temp_list = []
    while len(temp_list) < 10:
        ret, frame = video.read()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        disp_name = vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.5
            )
        print(disp_name) #debugging
        disp_name = disp_name.split(":")[0]
        print(disp_name)

        num_result = 0

        if disp_name == "bicycle":
            num_result = a_dict["bicycle"]
        elif disp_name == "child":
            num_result = a_dict["child"]
        elif disp_name == "const":
            num_result = a_dict["const"]
        elif disp_name == "bump":
            num_result = a_dict["bump"]
        elif disp_name == "cross":
            num_result = a_dict["cross"]
        elif disp_name == "parking":
            num_result = a_dict["parking"]    
        elif disp_name == "bus":
            num_result = a_dict["bus"]
        elif disp_name == "left":
            num_result = a_dict["left"]
        elif disp_name == "right":
            num_result = a_dict["right"]            
            # print(temp_list)
        temp_list.append(num_result)

    #----------------------------------------------------------

    print(temp_list)
    num_1 = temp_list.count(1)
    num_2 = temp_list.count(2)
    num_3 = temp_list.count(3)
    num_4 = temp_list.count(4)
    num_5 = temp_list.count(5)
    num_6 = temp_list.count(6)
    num_7 = temp_list.count(7)
    num_8 = temp_list.count(8)
    num_9 = temp_list.count(9)


    result = 0
    if num_1 >= 3:
        result = 1
    elif num_2 >= 3:
        result = 2
    elif num_3 >= 3:
        result = 3
    elif num_4 >= 3:
        result = 4
    elif num_5 >= 3:
        result = 5
    elif num_6 >= 3:
        result = 6
    elif num_7 >= 3:
        result = 7
    elif num_8 >= 3:
        result = 8
    elif num_9 >= 3:
        result = 9
    else:
        print("nothing! result will be 0")

    print(result)
    str_result = str(result)
    data = bytes(str_result, encoding='utf-8')
    send_sock.sendto(data, dest)

    cv2.imshow('object',frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()


