from __future__ import print_function
import sys
import openvino
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore
import socket

def build_argparser():
    """
    what parser do we NEED?
    - model: xml and bin file
    - device: default CPU. But check GPU later
    - number_top: optional
    """
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      default="./model.xml", type=str)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; "
                           "CPU, GPU, FPGA or MYRIAD is acceptable. "
                           "Sample will look for a suitable plugin for device specified (CPU by default)",
                      default="CPU", type=str)
    return parser


def main():
    args = build_argparser().parse_args()
    ie = IECore()
    # --------------------------- 1. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    net = ie.read_network(model=model_xml, weights=model_bin)
    #-----socket UDP----
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = ("localhost", 9999)
    # --------------------------- 3. Read and preprocess input --------------------------------------------

    for input_key in net.input_info:
        print("input shape: " + str(net.input_info[input_key].input_data.shape))
        print("input key: " + input_key)
        if len(net.input_info[input_key].input_data.layout) == 4:
            n, c, h, w = net.input_info[input_key].input_data.shape # n=1, c=3, h=300, w=300

    #----------------------START-------------------------
    video = cv2.VideoCapture(0)
    while(True):
        temp_list = []
        while len(temp_list) < 5:
            images = np.ndarray(shape=(n, c, h, w))
            images_hw = []
            ret, frame = video.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = frame_rgb
            ih, iw = image.shape[:-1]
            images_hw.append((ih, iw))

            # RESIZE
            if (ih, iw) != (h, w):
                image = cv2.resize(image, (w, h))
            draw_image = image
            image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            images[0] = image #i=0

            # --------------------------- 4. Configure input & output ---------------------------------------------
            # --------------------------- Prepare input blobs -----------------------------------------------------
            assert (len(net.input_info.keys()) == 1 or len(
                net.input_info.keys()) == 2), "Sample supports topologies only with 1 or 2 inputs"
            out_blob = next(iter(net.outputs))
            input_name, input_info_name = "", ""

            for input_key in net.input_info:
                if len(net.input_info[input_key].layout) == 4:
                    input_name = input_key
                    log.info("Batch size is {}".format(net.batch_size))
                    net.input_info[input_key].precision = 'U8'
                elif len(net.input_info[input_key].layout) == 2:
                    input_info_name = input_key
                    net.input_info[input_key].precision = 'FP32'
                    if net.input_info[input_key].input_data.shape[1] != 3 and net.input_info[input_key].input_data.shape[1] != 6 or \
                        net.input_info[input_key].input_data.shape[0] != 1:
                        log.error('Invalid input info. Should be 3 or 6 values length.')

            data = {}
            data[input_name] = images

            if input_info_name != "":
                infos = np.ndarray(shape=(n, c), dtype=float)
                for i in range(n):
                    infos[i, 0] = h
                    infos[i, 1] = w
                    infos[i, 2] = 1.0
                data[input_info_name] = infos

            # --------------------------- Prepare output blobs ----------------------------------------------------
            # --------------------------- Performing inference ----------------------------------------------------
            exec_net = ie.load_network(network=net, device_name=args.device)
            res = exec_net.infer(inputs=data) # dictionary # length: 1
            # --------------------------- Read and postprocess output ---------------------------------------------
            res = res[out_blob]
            data = res[0][0][:,1:3] #(100, 7)
            #data = data[:,1:3] #(100, 2)
            data = data[np.argmax(data[:, 1]),:]
            print(data) # [2.         0.11747734]

            print("snap!")
            if data[1] > 0.6: # probability 0.5 이상
                print("!! I'm CONFINDENT !!")
                label = np.int(data[0])
                temp_list.append(label)
            else:
                temp_list.append(0) #아닐 때 0

        print("here: ",temp_list)
    
    #---make RESULT---
    num_1 = temp_list.count(1)
    num_2 = temp_list.count(2)
    num_3 = temp_list.count(3)
    num_4 = temp_list.count(4)
    num_5 = temp_list.count(5)
    num_6 = temp_list.count(6)
    num_7 = temp_list.count(7)
    num_8 = temp_list.count(8)
    num_9 = temp_list.count(9)
    num_10 = temp_list.count(10)

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
    elif num_10 >= 3:
        result = 10
    else:
        result = 0

    print("SENDING: ",result)
    #---socket---
    str_result = str(result)
    data = bytes(str_result, encoding='utf-8')
    send_sock.sendto(data, dest)

if __name__ == '__main__':
    sys.exit(main() or 0)
