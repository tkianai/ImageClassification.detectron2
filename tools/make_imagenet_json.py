'''
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-26 17:27:16
@FilePath       : /ImageCls.detectron2/tools/make_imagenet_json.py
@Description    : 
'''

import os
import os.path as osp
from PIL import Image
import argparse
import json
from tqdm import tqdm


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

ARCHIVE_META = {
    'train': 'ILSVRC2012_img_train',
    'val': 'ILSVRC2012_img_val',
}


def parse_args():
    
    parser = argparse.ArgumentParser(description="Make imagenet dataset d2-style")
    parser.add_argument('--root', type=str, help="ImageNet root directory")
    parser.add_argument('--save', type=str, help="Result saving directory")

    args = parser.parse_args()
    if not osp.exists(args.save):
        os.makedirs(args.save)
    
    assert osp.exists(osp.join(args.root, ARCHIVE_META['train']))
    assert osp.exists(osp.join(args.root, ARCHIVE_META['val']))

    return args


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def accumulate_imagenet_json(image_root):

    # accumulate infos
    classes = [d.name for d in os.scandir(image_root) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    dataset_dicts = []
    image_id = 1
    for target_class in tqdm(sorted(class_to_idx.keys())):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(image_root, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_image_file(path):
                    # NOTE PIl output order changed
                    # height, width = Image.open(path).size
                    # Be care for this: varies with different input
                    # Add
                    record = {
                        "file_name": osp.abspath(path),   # Using abs path, ignore image root, less flexibility
                        # "width": width,
                        # "height": height,
                        "image_id": image_id,
                        "label": class_index,
                        "class": target_class,
                    }
                    dataset_dicts.append(record)
                    image_id += 1

    return dataset_dicts, class_to_idx


def main(args):

    # Accumulate train
    dataset_dicts_train, class_to_idx = accumulate_imagenet_json(osp.join(args.root, ARCHIVE_META['train']))
    # Accumulate val
    dataset_dicts_val, _ = accumulate_imagenet_json(osp.join(args.root, ARCHIVE_META['val']))
    # Save
    with open(osp.join(args.save, "imagenet_detectron2_train.json"), "w") as w_obj:
        json.dump(dataset_dicts_train, w_obj)
    with open(osp.join(args.save, "imagenet_detectron2_val.json"), "w") as w_obj:
        json.dump(dataset_dicts_val, w_obj)
    with open(osp.join(args.save, "class_to_idx.json"), "w") as w_obj:
        json.dump(class_to_idx, w_obj)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
