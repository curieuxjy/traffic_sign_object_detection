# traffic_sign_object_detection
### Autonomous driving project TEAM PostNomad

![logo](/asset/logo.png)

## Process

![plan](/asset/plan.png)

0. Make traffic sign image database
1. Train a TF model(ssd_v2)
    * caution: it must be from 1.13.X version github respository
2. Make .xml .bim file for fast inferencing(Openvino)
    * openvino version: 20.
3. Send the results by UDP communication to LabView
    * it does not care about version or bits of program
    
## Experiment Log

* max_step 100000

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
    * download the fine_tune_model file after training
    * 
3. Send the results by UDP communication to LabView
