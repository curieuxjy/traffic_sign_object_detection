# traffic_sign_object_detection
### Autonomous driving project TEAM PostNomad

![logo](/asset/logo.png)

## Process

![plan](/asset/plan.png)

0. Make traffic sign image database
1. Train a TF model(ssd_v2)
    * it must be from `1.13.0` version [github respository](https://github.com/tensorflow/models/releases/tag/v1.13.0)
    * [Converting TensorFlow* Object Detection API Models](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html)
2. Make .xml .bim file for fast inferencing(Openvino)
    * openvino version: `2020.4`
    * visual studio version: `2019`
    * Cmake version: `3.18.1`
3. Send the results by UDP communication to LabView
    * it does not care about version or bits of program
    
## Experiment Log

* max_step 300000

|model_name|batch_size|learning_rate|decay_steps|decay_factor|momentum_optimizer_value|decay|epsilon|
|-|-|-|-|-|-|-|-|
|ssd_mobilenet_v2_coco|16|0.004|800720|0.95|0.9|0.9|1|
| | | | | | | | |

## Reference
* [1st repo](https://github.com/Tony607/object_detection_demo)
  * [blog](https://www.dlology.com/blog/how-to-train-an-object-detection-model-easy-for-free/)
  * [colaboratory](https://colab.research.google.com/github/Tony607/object_detection_demo/blob/master/tensorflow_object_detection_training_colab.ipynb)

* [2nd repo](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)
  * [youtube video](https://youtu.be/Rgpfk6eYxJA)
* [opencv set reference](https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html)
* [image processing](https://github.com/ethereon/lycon)
* [CPU, RAM check code](https://frhyme.github.io/python/python_check_memory_usage/)
* [OpenVino Model Optimizer](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Converting a TensorFlow* Model](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html#supported_topologies)


## Code Explanation along the process
0. Make traffic sign image database
1. Train a TF model(ssd_v2)
2. Make .xml .bim file for fast inferencing(Openvino)
    * Download the fine_tune_model file after training
    * Install [openvino](https://docs.openvinotoolkit.org/2020.4/openvino_docs_install_guides_installing_openvino_windows.html) and try some demos following the guidelines
    * or you can run [python sample code](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_ie_bridges_python_sample_object_detection_sample_ssd_README.html) `object_detection_sample_ssd.py` (`C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\inference_engine\samples\python\object_detection_sample_ssd`) 
    * Move to `C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\model_optimizer`
        * it is default directory
    * Make `temp` directory in the same location
    * Move training file(`frozen_inference_graph.pb`, `pipeline.config`) to `./temp`
    * Move `ssd_v2_support.json`file to `./temp`
        * original location: `C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\model_optimizer\extensions\front\tf`
    * Make `here` directory and outputs(`.xm`, `.bin`) will be there later
    * Run `mo_tf.py`
        ```
        python mo_tf.py --input_model ./temp/frozen_inference_graph.pb --output_dir ./here --transformations_config ./temp/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ./temp/pipeline.config
        ```
    * Get `.bin`, `.xml`, `.mapping` files
    * Test sample code with test img
        ```
        python object_detection_sample_ssd.py -m ./frozen_inference_graph.xml -i ./0.jpg
        ```
        * some details should be updated
        * sample output
        ```
        [0,1] element, prob = 0.993272    (0,0)-(298,238) batch id : 0 WILL BE PRINTED!
        [1,1] element, prob = 0.99045    (0,172)-(499,516) batch id : 0 WILL BE PRINTED!
        [2,1] element, prob = 0.978142    (154,174)-(606,482) batch id : 0 WILL BE PRINTED!
        [3,1] element, prob = 0.971454    (441,301)-(720,649) batch id : 0 WILL BE PRINTED!
        [4,1] element, prob = 0.956165    (344,182)-(720,500) batch id : 0 WILL BE PRINTED!
        [5,1] element, prob = 0.954187    (47,15)-(481,330) batch id : 0 WILL BE PRINTED!
        [6,1] element, prob = 0.946346    (0,151)-(347,517) batch id : 0 WILL BE PRINTED!
        [7,1] element, prob = 0.943107    (0,80)-(246,720) batch id : 0 WILL BE PRINTED!
        [8,1] element, prob = 0.942482    (0,317)-(310,676) batch id : 0 WILL BE PRINTED!
        ```
        
        Options
        ```
        usage: object_detection_sample_ssd.py [-h] -m MODEL -i INPUT [INPUT ...]
                                      [-l CPU_EXTENSION] [-d DEVICE]
                                      [--labels LABELS] [-nt NUMBER_TOP]

        Options:
          -h, --help            Show this help message and exit.
          -m MODEL, --model MODEL
                                Required. Path to an .xml file with a trained model.
          -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                                Required. Path to image file.
          -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                                Optional. Required for CPU custom layers. Absolute
                                path to a shared library with the kernels
                                implementations.
          -d DEVICE, --device DEVICE
                                Optional. Specify the target device to infer on; CPU,
                                GPU, FPGA or MYRIAD is acceptable. Sample will look
                                for a suitable plugin for device specified (CPU by
                                default)
          --labels LABELS       Optional. Labels mapping file
          -nt NUMBER_TOP, --number_top NUMBER_TOP
                                Optional. Number of top results
        ```
3. Send the results by UDP communication to LabView
