import albumentations as album


def train_transform() -> album.Compose:
    """A functions to transform train data
    1. Vertical Frip (0.5)
    2. Rotation ([-10, 10])
    3. Cut out from the center
    4. Cut out randomly
    """
    return album.Compose([
        album.VerticalFlip(p=0.5),
        album.Rotate(limit=[-10, 10]),
        album.CenterCrop(height=512, width=256),
        album.RandomCrop(height=224, width=224),
    ])


def val_transform() -> album.Compose:
    """A functions to transform validate data
    1. Cut out from the center
    """
    return album.Compose([
        album.CenterCrop(height=224, width=224),
    ])
