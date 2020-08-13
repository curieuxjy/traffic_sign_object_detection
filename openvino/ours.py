from __future__ import print_function
import sys
import openvino
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore
# import pprint
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
#%matplotlib inline


def build_argparser():
    """
    what parser do we NEED?
    - model: xml and bin file
    - input: input image. it will be capture image from video camera
    - device: default CPU. But check GPU later
    - number_top: optional
    """
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    # args.add_argument("-i", "--input", help="Required. Path to image file.",required=True, type=str, nargs="+")
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; "
                           "CPU, GPU, FPGA or MYRIAD is acceptable. "
                           "Sample will look for a suitable plugin for device specified (CPU by default)",
                      default="CPU", type=str)
    # args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)
    return parser


def main():
    #log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    #log.info("Loading Inference Engine")
    ie = IECore()
    # --------------------------- 1. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    # log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)
    # xml bin 파일 읽고 확인
    # -----------------------------------------------------------------------------------------------------

    # ------------- 2. Load Plugin for inference engine and extensions library if specified --------------
    # log.info("Device info:")
    # versions = ie.get_versions(args.device)
    # print("{}{}".format(" " * 8, args.device)) 
    # print("{}MKLDNNPlugin version ......... {}.{}".format(" " * 8, versions[args.device].major,
    #                                                       versions[args.device].minor))
    # print("{}Build ........... {}".format(" " * 8, versions[args.device].build_number))

    # we do not have Not supported layer
    # -----------------------------------------------------------------------------------------------------

    # --------------------------- 3. Read and preprocess input --------------------------------------------

    # print("inputs number: " + str(len(net.input_info.keys())))

    for input_key in net.input_info:
        print("input shape: " + str(net.input_info[input_key].input_data.shape))
        print("input key: " + input_key)
        if len(net.input_info[input_key].input_data.layout) == 4:
            n, c, h, w = net.input_info[input_key].input_data.shape # n=1, c=3, h=300, w=300
            # print("n: ", n)
            # print("c: ", c)
            # print("h: ", h)
            # print("w: ", w)


    #----------------------START-------------------------
    # a_dict = {"bicycle": 1, "child":2, "const":3, "bump":2, "cross":4, "inter":5, "parking":6, "bus":7, "left":8, "right":9}
    video = cv2.VideoCapture(0)
    while(True):
        temp_list = []
        while len(temp_list) < 5:
            images = np.ndarray(shape=(n, c, h, w))
            images_hw = []
            #for i in range(n): # n=1 just once
            ret, frame = video.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = frame_rgb
            # frame_expanded = np.expand_dims(frame_rgb, axis=0)

            # image = cv2.imread(args.input[i])
            ih, iw = image.shape[:-1]
            images_hw.append((ih, iw))
            # log.info("File was added: ")
            # log.info("        {}".format(args.input[i]))

            # RESIZE
            if (ih, iw) != (h, w):
                #log.warning("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
                image = cv2.resize(image, (w, h))
            # cv2.imshow('image', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            draw_image = image
            image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            images[0] = image #i=0
                

            # -----------------------------------------------------------------------------------------------------

            # --------------------------- 4. Configure input & output ---------------------------------------------
            # --------------------------- Prepare input blobs -----------------------------------------------------
            # log.info("Preparing input blobs")
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
            # log.info('Preparing output blobs')

            output_name, output_info = "", net.outputs[next(iter(net.outputs.keys()))]
            for output_key in net.outputs:
                if net.layers[output_key].type == "DetectionOutput":
                    output_name, output_info = output_key, net.outputs[output_key]

            if output_name == "":
                log.error("Can't find a DetectionOutput layer in the topology")

            output_dims = output_info.shape
            if len(output_dims) != 4:
                log.error("Incorrect output dimensions for SSD model")
            max_proposal_count, object_size = output_dims[2], output_dims[3]

            if object_size != 7:
                log.error("Output item should have 7 as a last dimension")

            output_info.precision = "FP32"
            # -----------------------------------------------------------------------------------------------------

            # --------------------------- Performing inference ----------------------------------------------------
            # log.info("Loading model to the device")
            exec_net = ie.load_network(network=net, device_name=args.device)
            # log.info("Creating infer request and starting inference")
            res = exec_net.infer(inputs=data) # dictionary # length: 1
            # print("------------------------result-----------------------")
            # pp = pprint.PrettyPrinter(width=20, indent=4)
            # pp.pprint(res)
            # {   'DetectionOutput': array([[[[0.00000000e+00, 1.00000000e+00, 1.97501704e-02,
            #       0.00000000e+00, 2.46290386e-01, 9.84473228e-01,
            #       9.66810405e-01],
            #      [0.00000000e+00, 1.00000000e+00, 3.67655582e-03,
            #       2.67853945e-01, 0.00000000e+00, 9.82620358e-01,
            #       7.01416850e-01],

            # -----------------------------------------------------------------------------------------------------

            # --------------------------- Read and postprocess output ---------------------------------------------
            # log.info("Processing output blobs")
            temp = res
            res = res[out_blob]
            # (1, 1, 100, 7)
            # 0: batch index
            # 1: class label
            # 2: class probability
            # 3: x_1 box coordinate (0~1 as a fraction of the image width reference to the upper left corner)
            # 4: y_1 box coordinate (0~1 as a fraction of the image height reference to the upper left corner)
            # 5: x_2 box coordinate (0~1 as a fraction of the image width reference to the upper left corner)
            # 6: y_2 box coordinate (0~1 as a fraction of the image height reference to the upper left corner)

            boxes, classes = {}, {}
            data = res[0][0] #(100, 7)
            # print("-------data---------")
            #print(data)
            print("view speed")
            for number, proposal in enumerate(data):
                if proposal[2] > 0: # probability 0 이상
                    imid = np.int(proposal[0])
                    #ih, iw = images_hw[imid]
                    label = np.int(proposal[1])
                    confidence = proposal[2]
                    # xmin = np.int(iw * proposal[3])
                    # ymin = np.int(ih * proposal[4])
                    # xmax = np.int(iw * proposal[5])
                    # ymax = np.int(ih * proposal[6])
                    # print("[{},{}] element, prob = {:.6}    ({},{})-({},{}) batch id : {}" \
                    #       .format(number, label, confidence, xmin, ymin, xmax, ymax, imid), end="")
                    if proposal[2] > 0.5: # probability 0.5 이상
                        print(" WILL BE PRINTED!")
                        # if not imid in boxes.keys():
                        #     boxes[imid] = []
                        # boxes[imid].append([xmin, ymin, xmax, ymax])
                        # if not imid in classes.keys():
                        #     classes[imid] = []
                        # classes[imid].append(label)
                        temp_list.append(proposal[1])

                    else:
                        #print()
                        pass
        print(temp_list)

    # Deploy
    # probability_threshold = 0.5
    # #print(temp[out_blob])
    # preds = [pred for pred in temp[out_blob][0][0] if pred[2] > probability_threshold]
    # ax= plt.subplot(1, 1, 1)
    # #plt.imshow(draw_image)  # slice by z axis of the box - box[0].
    # for pred in preds:
    #     class_label = pred[1]
    #     probability = pred[2]
    #     # print('Predict class label:{:.0f}, with probability: {:.2f}'.format(
    #     #     class_label, probability))
    #     box = pred[3:]
    #     box = (box * np.array(image.shape[:2][::-1] * 2)).astype(int)
    #     x_1, y_1, x_2, y_2 = box
    #     rect = patches.Rectangle((x_1, y_1), x_2-x_1, y_2 -
    #                             y_1, linewidth=2, edgecolor='red', facecolor='none')
    #     ax.add_patch(rect)
    #     ax.text(x_1, y_1, '{:.0f} - {:.2f}'.format(class_label,
    #                                             probability), fontsize=12, color='yellow')
    # fig.show()

    # ORIGINAL
    # for imid in classes:
    #     print(type(image))
    #     tmp_image = cv2.UMat(image)
    #     for box in boxes[imid]:
    #         cv2.rectangle(tmp_image, (box[0], box[1]), (box[2], box[3]), (232, 35, 244), 2)
    #     cv2.imwrite("out.bmp", tmp_image)
    #     log.info("Image out.bmp created!")
    #     print("????????")
    # -----------------------------------------------------------------------------------------------------

    #log.info("Execution successful\n")
    # log.info(
    #     "This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool")


if __name__ == '__main__':
    sys.exit(main() or 0)
