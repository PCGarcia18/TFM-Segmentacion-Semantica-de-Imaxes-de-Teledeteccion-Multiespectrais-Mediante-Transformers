# Mask2Former for multispectral images
This repository contains the adaptions made to Mask2Former to handle multispectral data.


Mask2Former paper: https://arxiv.org/pdf/2112.01527.pdf  
Mask2Former repository: https://github.com/facebookresearch/Mask2Former  
Detectron2 repository: https://github.com/facebookresearch/detectron2  
Interactive demo: https://pcgarcia18.github.io/image_test_tfm/  
  
## Segmentation examples

![Cool lakes](cool-lakes.gif)
![Cool river](cool-river.gif)
![Cool big city](cool-big-city.gif)
![Cool city near river](cool-city-near-river.gif)


## Installing Detectron2 and Mask2Former
Env creation:
* conda create --name mask2former python=3.8 -y
* conda activate mask2former

Pytorch and Opencv installation:
* conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
* pip install -U opencv-python

Detectron2 setup:
* git clone git@github.com:facebookresearch/detectron2.git
* cd detectron2
* pip install -e .
* pip install git+https://github.com/cocodataset/panopticapi.git
* pip install git+https://github.com/mcordts/cityscapesScripts.git

Mask2former setup:
* cd ..
* git clone git@github.com:facebookresearch/Mask2Former.git
* cd Mask2Former
* pip install -r requirements.txt
* cd mask2former/modeling/pixel_decoder/ops
* sh make.sh

Working folder:
* cd ~/wip

### Resulting folder structure after installing Detectron2, Mask2Former and creating the dataset

WIP/  
├── detectron2  
└── Mask2former    

##  Creating the dataset and modifying Detectron2 and Mask2Former for multispetral images

Cloning the repository: 
* cd ~/wip
* git clone git@gitlab.citius.gal:hiperespectral/mask2former-for-multispectral-images.git 

Creating the dataset:
* Download from: https://x-ytong.github.io/project/Five-Billion-Pixels.html
* Use ./code/dataset_rawb_creation.ipynb to create the images in the correct format

Copying the modified files to Detectron2 and Mask2Former:

* cp mask2former-for-multispectral-images/code/dataset_evaluator/sem_seg_evaluation_RAWB.py detectron2/detectron2/evaluation
* cp mask2former-for-multispectral-images/code/dataset_mappers/dataset_mapper.py detectron2/detectron2/data
* cp mask2former-for-multispectral-images/code/dataset_mappers/mask_former_semantic_dataset_mapper_RAWB.py Mask2Former/mask2former/data/dataset_mappers
* cp mask2former-for-multispectral-images/code/utils/detection_utils.py /home/pablo.canosa/wip/detectron2/detectron2/data
* cp mask2former-for-multispectral-images/code/train_net_gf_8bit_rawb_small.py Mask2Former
* cp mask2former-for-multispectral-images/code/demo_rawb.py Mask2Former/demo

* The init.py files need to be modified to correctly use the mappers and the evaluator.




### Resulting folder structure after adding the files for multispectral

WIP/  
├── datasets/  
│   ├── train/  
│   │   ├── gaofen_train_images/  
│   │   │   ├── img_1.rawb  
│   │   │   ├── _____.rawb  
│   │   │   └── img_480.rawb  
│   │   └── gaofen_test_png_GT  
│   └── test/  
│       ├── gaofen_test_images/  
│       │   ├── img_1.rawb  
│       │   ├── _____.rawb  
│       │   └── img_120.rawb  
│       └── gaofen_test_png_GT   
├── detectron2/  
│   └── detectron2/  
│       ├── data/  
│       │   ├── detection_utils.py (REPLACED)  
│       │   └── dataset_mapper.py (REPLACED)  
│       └── evaluation/  
│           └── sem_seg_evaluation_RAWB.py (ADDED)  
├── Mask2former/  
│   ├── mask2former/  
│   │   └── data/  
│   │       └── datasets_mapper/  
│   │           └── mask_former_semantic_dataset_mapper_RAWB.py (ADDED)  
│   ├── demo/  
│   │   └── demo_rawb.py (ADDED)  
│   └── train_net_gf_8bit_rawb_small.py (ADDED)  
└── mask2former-for-multispectral/  


## Code usage  

### Training  
* python train_net_gf_8bit_rawb_small.py --num-gpus 1 --config-file /path_to_config/config.yaml

### Demo  

* python demo_rawb.py --config-file /path_to_config/config.yaml --input /path_to_image/img.rawb --opts MODEL.WEIGHTS /path_to_weights/weights.pth  MODEL.MASK_FORMER.TEST.INSTANCE_ON False MODEL.MASK_FORMER.TEST.PANOPTIC_ON False


