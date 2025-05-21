import os
import random
import string
import tempfile
import numpy as np

from PIL import Image


def generate_random_filename(extension: str = ""):
    temp_dir = tempfile.gettempdir()
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    extension = f".{extension}" if extension else ""
    return os.path.join(temp_dir, f"tmpl{random_string}{extension}")


def generate_output_filename(base_name: str, model_name: str, extension: str = ""):
    temp_dir = tempfile.gettempdir()
    extension = f".{extension}" if extension else ""
    name = f"{base_name}_{model_name}" if model_name else base_name
    output_filename = os.path.join(temp_dir, f"{name}{extension}")

    def create_non_existing_filename(discriminator: int = None):
        discriminator = discriminator or 1
        output_filename = os.path.join(temp_dir, f"{name}_{discriminator}{extension}")
        if not os.path.exists(output_filename):
            return output_filename
        else:
            return create_non_existing_filename(discriminator + 1)

    if os.path.exists(output_filename):
        output_filename = create_non_existing_filename()

    return output_filename


def pad_image(image_array, slice_height, slice_width):
    pad_height = slice_height - image_array.shape[0] % slice_height
    pad_width = slice_width - image_array.shape[1] % slice_width
    padded_image = np.pad(image_array, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
    return padded_image, pad_height, pad_width


def slice_image(image_array, slice_height, slice_width):
    slices = []
    padded_image, pad_height, pad_width = pad_image(image_array, slice_height, slice_width)
    for i in range(0, padded_image.shape[0], slice_height):
        for j in range(0, padded_image.shape[1], slice_width):
            slice_ = padded_image[i : i + slice_height, j : j + slice_width]
            slices.append(slice_)
    return slices, pad_height, pad_width


def stitch_image(slices, original_height, original_width, slice_height, slice_width, pad_height, pad_width):
    padded_height = original_height + pad_height
    padded_width = original_width + pad_width
    stitched_image = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)  # Ensure 3 channels for RGB
    index = 0
    for i in range(0, padded_height, slice_height):
        for j in range(0, padded_width, slice_width):
            slice_ = slices[index]
            if slice_.ndim == 2 or slice_.shape[2] == 1:  # Check if slice_ is grayscale
                slice_ = np.stack((slice_,) * 3, axis=-1)  # Convert to RGB by stacking
            mask = slice_ != 0
            stitched_image[i : i + slice_height, j : j + slice_width][mask] = slice_[mask]
            index += 1
    return stitched_image[:original_height, :original_width]


if __name__ == '__main__':
    input_file = "ai_inference/test_images/Plot_4_.png"
    with Image.open(input_file).convert("RGB") as input_image:
        image_array = np.array(input_image)

    slices, pad_height, pad_width = slice_image(image_array, 3200, 480)
    stitched_image = stitch_image(slices, image_array.shape[0], image_array.shape[1], 3200, 480, pad_height, pad_width)

    with Image.fromarray(stitched_image.astype(np.uint8), mode='RGB') as image_pil:
        result_image = np.array(image_pil)

    print("Done")
