# traffic_sign_object_detection
### Autonomous driving project TEAM PostNomad

![logo](/asset/logo.png)

## Plan

![plan](/asset/plan.png)

## Experiment Log

* max_step 100000

|model_name|batch_size|learning_rate|decay_steps|decay_factor|momentum_optimizer_value|decay|epsilon|
|-|-|-|-|-|-|-|-|
|ssd_mobilenet_v2_coco|16|0.004|800720|0.95|0.9|0.9|1|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

# problem 
* [이미지 처리](https://github.com/ethereon/lycon)
* [CPU, RAM 사용 확인](https://frhyme.github.io/python/python_check_memory_usage/)
* [Tensorflow Lite](https://www.tensorflow.org/lite/guide/inference?hl=ko#load_and_run_a_model_in_python)


## Reference
* [1st repo](https://github.com/Tony607/object_detection_demo)
  * [blog](https://www.dlology.com/blog/how-to-train-an-object-detection-model-easy-for-free/)
  * [colaboratory](https://colab.research.google.com/github/Tony607/object_detection_demo/blob/master/tensorflow_object_detection_training_colab.ipynb)

* [2nd repo](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)
  * [youtube video](https://youtu.be/Rgpfk6eYxJA)
* [opencv set reference](https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html)