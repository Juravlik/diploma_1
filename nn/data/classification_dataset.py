import os
import cv2
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from nn.data import _get_training_augmentation



class ClassificationDataset(Dataset):
    def __init__(
            self,
            root: str,
            path: str,
            class_to_label: dict,
            augmentation=None,
            preprocessing=None,
    ):

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.class_to_label = class_to_label
        self.images, self.labels = self._create_dataset(root, path)

    @staticmethod
    def _create_dataset(root, path):
        images = []
        labels = []

        for root, dirs, _ in os.walk(os.path.join(root, path)):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                for _, _, files in os.walk(dir_path):
                    for file in files:
                        images.append(os.path.join(dir_path, file))
                        labels.append(dir)

        assert len(images) == len(labels)

        return images, labels

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentation:
            image = self.augmentation(image=image)['image']

        if self.preprocessing:
            image = self.preprocessing(image=image)['image']

        label = self.class_to_label[self.labels[index]]

        return {
            'image': image,
            'label': label
        }

    def __len__(self):
        return len(self.images)

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)


if __name__ == '__main__':

    class_to_label = dict()
    dirs = os.listdir(os.path.join('/home/juravlik/PycharmProjects/diploma_1/static/data/', 'train'))
    for i, d in enumerate(dirs):
        class_to_label[d] = i

    train_dataset = ClassificationDataset('/home/juravlik/PycharmProjects/diploma_1/static/data/', 'train',
                                          class_to_label=class_to_label, augmentation=_get_training_augmentation())
    train_loader = train_dataset.get_dataloader(batch_size=1, shuffle=False)

    for data in train_loader:
        print(data['label'])
        plt.imshow(data['image'][0].numpy(), cmap='gray')
        plt.show()
