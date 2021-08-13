import os
from torchvision import datasets
from torchvision.transforms import transforms

ImageNetNormalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

ImageNetTrainTransform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ImageNetNormalize,
])

ImageNetValidationTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ImageNetNormalize,
])

ImageNetTestTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.Pad(57, padding_mode="symmetric"),
    transforms.RandomRotation((0, 0)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ImageNetNormalize,
])


# train dataset example for image-net
def train_dataset(data_dir, transform=ImageNetTrainTransform):
    train_dir = os.path.join(data_dir, 'train')
    return datasets.ImageFolder(train_dir, transform)


# val dataset example for image-net
def val_dataset(data_dir, transform=ImageNetValidationTransform):
    val_dir = os.path.join(data_dir, 'val')
    return datasets.ImageFolder(val_dir, transform)


# test dataset example for image-net
def test_dataset(data_dir, transform=ImageNetTestTransform):
    test_dir = os.path.join(data_dir, 'val')
    return datasets.ImageFolder(test_dir, transform)
