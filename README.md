# traffic_sign_object_detection
### Autonomous driving project TEAM PostNomad

![logo](/asset/logo.png)

## Process

![plan](/asset/plan.png)

0. Make traffic sign image database
1. Train a TF model(ssd_v2)
    * it must be from `1.13.0` version [github respository](https://github.com/tensorflow/models/releases/tag/v1.13.0)
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
* [이미지 처리](https://github.com/ethereon/lycon)
* [CPU, RAM 사용 확인](https://frhyme.github.io/python/python_check_memory_usage/)


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
    * Move `json`file to `./temp`
        * ``
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
        * some details 
3. Send the results by UDP communication to LabView
