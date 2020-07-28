import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def start():
    
    PATH_TO_CKPT = "D:/GitHub/traffic_sign_object_detection/fine_tuned_model/frozen_inference_graph.pb"
    PATH_TO_LABELS = "D:/GitHub/traffic_sign_object_detection/data/annotations/label_map.pbtxt"

    NUM_CLASSES = 5

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Initialize webcam feed
    video = cv2.VideoCapture(0) # 0:web_cam 1:logitech
    ret = video.set(3,720)
    ret = video.set(4,720)

    a_dict = {"bicycle": 1, "child":2, "const":3, "bump":2, "cross":4}
    #disp_name2 = "5"
    while(True):
        # temp_list = []
        # while len(temp_list) < 5:
        # start_1 = time.time() # 시작
        ret, frame = video.read()
        # print("------one-frame-------")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        # start_2 = time.time()
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        # print("sess.run 시간: ", time.time()-start_2)
        # print("-------display-visulalization-start-----")
        
        disp_name = vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.6
            )
            # name:percentage
            
        disp_name = disp_name.split(":")[0]
        #     print(disp_name)
        #     # print("한 프레임 판단 시간: ", time.time()-start_1)

        #     print("change string to number")
        #     num_result = 0
        #     if disp_name == "bicycle":
        #         num_result = a_dict["bicycle"]
        #     elif disp_name == "child":
        #         num_result = a_dict["child"]
        #     elif disp_name == "const":
        #         num_result = a_dict["const"]
        #     elif disp_name == "bump":
        #         num_result = a_dict["bump"]
        #     elif disp_name == "cross":
        #         num_result = a_dict["cross"]
        #     temp_list.append(num_result)

        # print(temp_list)
        # num_1 = temp_list.count(1)
        # num_2 = temp_list.count(2)
        # num_3 = temp_list.count(3)
        # num_4 = temp_list.count(4)
        # # num_5 = temp_list.count(5)

        # result = 0
        # if num_1 >= 3:
        #     result = 1
        # elif num_2 >= 3:
        #     result = 2
        # elif num_3 >= 3:
        #     result = 3
        # elif num_4 >= 3:
        #     result = 4
        # else:
        #     print("nothing! result will be 0")

        # str_result = str(result)

        # # with open("result.txt", "w") as f:
        # #     f.write(str(result))

        # # time.sleep(5)

        # # All the results have been drawn on the frame, so it's time to display it.
        # # cv2.imshow('Object detector', frame)

        # # Press 'q' to quit
        # if cv2.waitKey(1) == ord('q'):

        if disp_name!=False:
            break

    # video.release()
    # cv2.destroyAllWindows()
    return disp_name