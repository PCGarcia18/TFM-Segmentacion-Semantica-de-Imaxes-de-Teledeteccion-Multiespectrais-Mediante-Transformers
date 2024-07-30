# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
# 

import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data import detection_utils as utils
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False





from detectron2.data import MetadataCatalog, DatasetCatalog
import gc

if __name__ == "__main__":

#######
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
    thing_dataset_id_to_contiguous_id = {i: i for i in range(25)}
    # Gaofen TRAIN
    MetadataCatalog.get("gaofen_train").stuff_classes = ALL_CLASSES
    MetadataCatalog.get("gaofen_train").stuff_colors = COLOR_LIST



    MetadataCatalog.get("gaofen_train").thing_dataset_id_to_contiguous_id = thing_dataset_id_to_contiguous_id


    # Gaofen TEST

    MetadataCatalog.get("gaofen_test").stuff_classes = ALL_CLASSES  
    MetadataCatalog.get("gaofen_test").stuff_colors = COLOR_LIST

    MetadataCatalog.get("gaofen_test").thing_dataset_id_to_contiguous_id = thing_dataset_id_to_contiguous_id


#######
#######
####### 


    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            
            # carga de imagenes en rawb

            img = utils.read_rawb_NirRGB(path)

            img = img[:, :, ::-1] # in the visualizer the images are read backwards
            
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            print(predictions["sem_seg"].shape)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.imwrite("panoptic.jpg",visualized_output.get_image()[:, :, ::-1])
                
                mask_path = path
                file_name_with_extension = os.path.basename(mask_path)
                file_name, file_extension = os.path.splitext(file_name_with_extension)
                
                # Shows the image with opencv
                
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                
                if cv2.waitKey(0) == 27:
                    break  # esc to quit

                ####
                # EXTRA CODE TO TINKER WITH THE OUTPUT
                ####

                #path_color_masks = "/home/pablo.canosa/wip/datasets/small_gaofen/test/test_masks_tif/"

                #color_mask = cv2.imread(f"{path_color_masks}{file_name}_24label.tif", cv2.IMREAD_UNCHANGED)
                

                """
                dest_path = "rawb"

                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                
                # Creates a concatenated image with the folowing images: ground truth, result and original
                result = np.concatenate((color_mask, visualized_output.get_image()[:, :, ::-1]), axis=1)
                result = np.concatenate((result, img[:,:,0:3] ), axis=1)
                cv2.imwrite(f"{dest_path}/{file_name}_result.jpg", result)
                """
                
                # Saves the color mask, the original and the result in the same folder
                """
                if not os.path.exists(f"{dest_path}/{file_name}"):
                    os.makedirs(f"{dest_path}/{file_name}")
                cv2.imwrite(f"{dest_path}/{file_name}/image.jpg", img[:,:,0:3])
                cv2.imwrite(f"{dest_path}/{file_name}/mask.jpg", color_mask)
                cv2.imwrite(f"{dest_path}/{file_name}/result.jpg", visualized_output.get_image()[:, :, ::-1])
                """
                
                
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
