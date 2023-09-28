import os
from typing import List


def load_yolo_dataset(dataset_path, split) -> List[str]:
    images_dir = os.path.join(dataset_path, split, 'images')
    images_names = os.listdir(images_dir)
    file_names = [file_name.rstrip('.jpg') for file_name in images_names]
    return file_names
