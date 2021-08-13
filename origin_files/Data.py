from turtle import pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

train_hd = pd.read_csv(
    '/home/anna/Desktop/workspace/Python/result_demo/train/train/train_labels.csv')  # 获取图片的名字我的csv文件储存在这里
train_path = '/home/anna/Desktop/workspace/Python/result_demo/train/train/images'  # 获取图片的路径


class Dataset(torch.utils.Data):

    def __init__(self, root, resize, mode):
        super(Dataset, self).__init__()

        self.root = root
        self.resize = resize

        if mode == 'train':  # 60%
            self.images = self.images[:int(0.6 * len(self.images))]
        elif mode == 'val':  # 20% = 60%->80%
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
        else:  # 20% = 80%->100%
            self.images = self.images[int(0.8 * len(self.images)):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)
        return img, label


def main():
    import time
    db = Dataset('Dataset', 64, 'train')

    x, y = next(iter(db))

    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)

    for x, y in loader:
        time.sleep(10)


if __name__ == '__main__':
    main()
