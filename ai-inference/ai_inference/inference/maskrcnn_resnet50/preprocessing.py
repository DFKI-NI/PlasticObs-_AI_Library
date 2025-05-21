# libraries
from PIL import Image
from torchvision import transforms as T


def resize(img, size=(1000, 1000)):
    """Resize an image

    Args:
        img (PIL.Image): Image loaded all pillow image
        size (tuple, optional): New size for the images. Defaults to (800, 800).

    Returns:
        PIL.Image: resized image
    """
    # Resize image
    img = img.resize(size, Image.Resampling.BILINEAR)

    return img


def get_transform(train):
    """Function to create transform object

    Args:
        train (bool): if the transform is used for training

    Returns:
        torch.Compose: Compose Object for transformations
    """
    transforms = []
    transforms.append(T.ToTensor())

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)
