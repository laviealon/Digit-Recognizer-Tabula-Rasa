"""This file is responsible for the processes of extracting the training data
from the directory <training_data> and transforming it into data that can be
managed by the file <data_manager.py>. The methods used are completely
non-standard as the purpose of this file is solely for the specific needs of the
<Digit-Recognizer-Tabula-Rasa> project."""
import numpy as np
from PIL import Image
from typing import List


def convert(img_filename: str) -> np.array:
    """Convert a 28 x 28 pixel image into a list of floats between 0 and
    1 with each element uniquely representing one pixel's greyscale value.
    """
    layer = []
    img = Image.open(img_filename)
    pix = img.load()
    for y in range(28):
        for x in range(28):
            greyscale_val = pix[x, y] / 255
            layer.append(greyscale_val)
    return np.array(layer)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
