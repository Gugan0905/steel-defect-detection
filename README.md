# steel-defect-detection
Made on the mmdetection framework.
https://github.com/open-mmlab/mmdetection

Dataset:
Severstal dataset
https://www.kaggle.com/c/severstal-steel-defect-detection/data


Cascade RCNN - 32x4d - ResNeXt101
Customized Generic RoI Extractor
https://arxiv.org/abs/2004.13665
Experimentally tuned hyperparameters
Custom configuration of Albumentation transformations in a repeated HSV transform scheme
https://github.com/albumentations-team/albumentations

![image](https://user-images.githubusercontent.com/43416760/123741308-705a7500-d8c7-11eb-8473-0f99a81a95cf.png)
a)

![image](https://user-images.githubusercontent.com/43416760/123741325-79e3dd00-d8c7-11eb-8831-29bc8dd77b9a.png)
b)

Fig 1. a) Before and b) After the repeated Hue Saturation value transform


![image](https://user-images.githubusercontent.com/43416760/123741364-8c5e1680-d8c7-11eb-997e-6dfe513217fc.png)
Fig 2. GRoIE layer architecture 1. RoI Pooling 2. Individual pre-processing 3. Feature aggregation 4. Attention module post-processing


Parameters	Description:

Pretrained Weight	COCO
Training scales	[(1333, 800), (1666,1000)]
Testing scales	[(1333, 800), (1666,1000)]
Validation scales	[(1333, 200), (1999,1000)]
Batch size	64
Optimizer	Stochastic Gradient Descent (SGD)
Learning Rate	0.001
Weight Decay	0.0001

![image](https://user-images.githubusercontent.com/43416760/123742609-941eba80-d8c9-11eb-8e6f-e06c0684e0d9.png)

________________________________________________________________________________________________________________
