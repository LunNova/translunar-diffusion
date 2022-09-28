import PIL
from PIL import Image, PngImagePlugin
import numpy as np

# beeeeeg
Image.MAX_IMAGE_PIXELS = 93312000000
# * 10 defaults
PngImagePlugin.MAX_TEXT_CHUNK = 10485760
PngImagePlugin.MAX_TEXT_MEMORY = 671088640


def load_image(
    image_file: str, crop: bool, size: int, interpolation, transform
):
    try:
        image = Image.open(image_file)
        if image.mode in ('P', 'L', 'RGB') and isinstance(
            image.info.get('transparency'), bytes
        ):
            # Convert images with transparency color set in bytes
            # to RGBA so transparency can be handled properly
            # fixes UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
            image = image.convert('RGBA')
        if image.mode == 'RGBA':
            # paste on white background if RGBA image
            # as .convert will leave artifacts
            image = Image.alpha_composite(
                Image.new('RGBA', image.size, (255, 255, 255)), image
            )
        if not image.mode == 'RGB':
            image = image.convert('RGB')
    except (OSError, ValueError) as e:
        print(f'Error with {image_file}', type(e), e)
        raise

    # default to score-sde preprocessing
    if crop and image.size[0] != image.size[1]:
        img = np.array(image).astype(np.uint8)
        crop_dim = min(img.shape[0], img.shape[1])
        h, w, = (
            img.shape[0],
            img.shape[1],
        )
        img = img[
            (h - crop_dim) // 2 : (h + crop_dim) // 2,
            (w - crop_dim) // 2 : (w + crop_dim) // 2,
        ]
        image = Image.fromarray(img)
        del img

    if size is not None and image.size != (size, size):
        image = image.resize((size, size), resample=interpolation)

    if transform:
        image = transform(image)

    np_image = np.array(image).astype(np.uint8)
    np_image = (np_image / 127.5 - 1.0).astype(np.float32)
    return np_image
