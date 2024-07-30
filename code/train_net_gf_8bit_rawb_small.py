# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified Mask2Former Training Script.

This script is a simplified version of the training script in detectron2/tools. Is now adapted to Multispectral data.
"""
USE_NIR_BAND = True # Set to True if you want to use the NIR band in the multispectral images, else it will train on RGB images

# The images have to be created from the Five Billion Pixels with the jupyter notebook provided in the repository


#Train images path, use your own path
TRAIN_IMAGES_PATH = '/home/pablo.canosa/wip/datasets/small_gaofen/train/8bit_rawb/' 
TRAIN_PNG_MASKS_PATH = '/home/pablo.canosa/wip/datasets/small_gaofen/train/png_masks/'

#Test images path, use your own path
TEST_IMAGES_PATH = '/home/pablo.canosa/wip/datasets/small_gaofen/test/test_8bit_rawb/' 
TEST_PNG_MASKS_PATH = '/home/pablo.canosa/wip/datasets/small_gaofen/test/test_masks_png/'


try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    SemSegEvaluatorRAWB,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    MaskFormerSemanticDatasetMapperRAWB,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        """
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )"""
        if evaluator_type == "sem_seg_RAWB":
            evaluator_list.append(
                SemSegEvaluatorRAWB(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Semantic mapper for rawb images
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic_RAWB":
            mapper = MaskFormerSemanticDatasetMapperRAWB(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg

# Load GaoFen Dataset
import pandas as pd
import cv2
import numpy as np

ALL_CLASSES = [
    "unlabeled",
    "industrial area",
    "paddy field",
    "irrigated field",
    "dry cropland",
    "garden land",
    "arbor forest",
    "shrub forest",
    "park",
    "natural meadow",
    "artificial meadow",
    "river",
    "urban residential",
    "lake",
    "pond",
    "fish pond",
    "snow",
    "bareland",
    "rural residential",
    "stadium",
    "square",
    "road",
    "overpass",
    "railway station",
    "airport"
]

COLOR_LIST = [
    (0, 0, 0),       # unlabeled
    (200, 0, 0),     # industrial area
    (0, 200, 0),     # paddy field
    (150, 250, 0),   # irrigated field
    (150, 200, 150), # dry cropland
    (200, 0, 200),   # garden land
    (150, 0, 250),   # arbor forest
    (150, 150, 250), # shrub forest
    (200, 150, 200), # park
    (250, 200, 0),   # natural meadow
    (200, 200, 0),   # artificial meadow
    (0, 0, 200),     # river
    (250, 0, 150),   # urban residential
    (0, 150, 200),   # lake
    (0, 200, 250),   # pond
    (150, 200, 250), # fish pond
    (250, 250, 250), # snow
    (200, 200, 200), # bareland
    (200, 150, 150), # rural residential
    (250, 200, 150), # stadium
    (150, 150, 0),   # square
    (250, 150, 150), # road
    (250, 150, 0),   # overpass
    (250, 200, 250), # railway station
    (200, 150, 0)    # airport

]

ID_TO_COLOR_DICT = {
    0: (0, 0, 0),
    1: (200, 0, 0),
    2: (0, 200, 0),
    3: (150, 250, 0),
    4: (150, 200, 150),
    5: (200, 0, 200),
    6: (150, 0, 250),
    7: (150, 150, 250),
    8: (200, 150, 200),
    9: (250, 200, 0),
    10: (200, 200, 0),
    11: (0, 0, 200),
    12: (250, 0, 150),
    13: (0, 150, 200),
    14: (0, 200, 250),
    15: (150, 200, 250),
    16: (250, 250, 250),
    17: (200, 200, 200),
    18: (200, 150, 150),
    19: (250, 200, 150),
    20: (150, 150, 0),
    21: (250, 150, 150),
    22: (250, 150, 0),
    23: (250, 200, 250),
    24: (200, 150, 0)
}

def get_gaofen_dict(images_path, gt_dir_png, gt_dir_tif_color): #Creates de dictionary with the information of the dataset in the Detectron2 format, gt_dir_tif_color is not yet used as is for panoptic segmentation
    
    dataset_dicts = []
    number_of_images = len(os.listdir(images_path))
    for image_idx, image_filename in enumerate(os.listdir(images_path)):

        print(f"{image_filename} is image {image_idx+1} out of {number_of_images}")
        record={}

        image_file_path = os.path.join(images_path, image_filename)  


        image_id, _= os.path.splitext(image_filename)

        gt_mask_grayscale = os.path.join(gt_dir_png, image_id + '_24label.png')
        record["sem_seg_file_name"] = gt_mask_grayscale

        # Mask size and image size is the same
        height, width = cv2.imread(gt_mask_grayscale, cv2.IMREAD_UNCHANGED).shape[:2]

        record["file_name"] = image_file_path
        record["image_id"] = image_id
        record["height"] = height
        record["width"] = width

        
        record["NIR"] = USE_NIR_BAND 

        #End loop and save dict
        dataset_dicts.append(record)

    return dataset_dicts

#######
from detectron2.data import MetadataCatalog, DatasetCatalog


def main(args):


    ### Register the datasets
    stuff_dataset_id_to_contiguous_id = {i: i for i in range(25)}#The dictionaries are trivial, the id is the same as the index
    
    # This are the paths for the dataset files in quadrants
    dataset_path_image = TRAIN_IMAGES_PATH
    dataset_path_png_mask = TRAIN_PNG_MASKS_PATH
    dataset_path_tif_mask = 'small_gaofen/train/tif_color_masks/' # Not used yet, for panoptic segmentation

    DatasetCatalog.register("gaofen_train", lambda : get_gaofen_dict(dataset_path_image,dataset_path_png_mask,dataset_path_tif_mask))
    MetadataCatalog.get("gaofen_train").stuff_classes = ALL_CLASSES
    MetadataCatalog.get("gaofen_train").ignore_label = 0
    MetadataCatalog.get("gaofen_train").thing_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
    MetadataCatalog.get("gaofen_train").stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
    
    MetadataCatalog.get("gaofen_train").stuff_colors = COLOR_LIST
    



    # test trials
    dataset_path_image_test = TEST_IMAGES_PATH
    dataset_path_png_mask_test = TEST_PNG_MASKS_PATH
    dataset_path_tif_mask_test = 'small_gaofen/test/test_masks_tif/'# Not used yet, for panoptic segmentation


    DatasetCatalog.register("gaofen_test", lambda : get_gaofen_dict(dataset_path_image_test,dataset_path_png_mask_test,dataset_path_tif_mask_test))
    MetadataCatalog.get("gaofen_test").stuff_classes = ALL_CLASSES 
    MetadataCatalog.get("gaofen_test").stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
    MetadataCatalog.get("gaofen_test").stuff_colors = COLOR_LIST

    MetadataCatalog.get("gaofen_test").ignore_label = 0 
    MetadataCatalog.get("gaofen_test").evaluator_type = "sem_seg_RAWB"


    ###
    
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()




if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
