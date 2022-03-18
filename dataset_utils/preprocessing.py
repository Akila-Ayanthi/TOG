from PIL import Image
import numpy as np


def letterbox_image_padded(image, size=(640, 640)):
    """ Resize image with unchanged aspect ratio using padding """
    image_copy = image.copy()
    print(np.asarray(image_copy))
    iw, ih = image_copy.size
    print(iw)
    print(ih)
    w, h = size
    scale = min(w / iw, h / ih)
    print(scale)
    nw = int(iw * scale)
    nh = int(ih * scale)
    print(nw)
    print(nh)

    image_copy = image_copy.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0, 0, 0))
    print(np.asarray(new_image))
    new_image.paste(image_copy, ((w - nw) // 2, (h - nh) // 2))
    print(new_image)
    new_image = np.asarray(new_image)[np.newaxis, :, :, :] / 255.
    print(new_image)
    meta = ((w - nw) // 2, (h - nh) // 2, nw + (w - nw) // 2, nh + (h - nh) // 2, scale)

    return new_image, meta
