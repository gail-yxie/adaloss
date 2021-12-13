from torchvision import transforms
from base import BaseDataLoader
from data_loader.cifar100 import get_cifar100
from parse_config import ConfigParser


class CIFAR100DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, training=True,
                 num_workers=4, pin_memory=True):
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']
        trans_type = cfg_trainer['trans_type']
        transform_train, transform_val = {}, {}

        transform_train['vit'] = transforms.Compose([
            # transforms.ColorJitter(brightness= 0.4, contrast= 0.4, saturation= 0.4, hue= 0.1),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_val['vit'] = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_train['res'] = transforms.Compose([
            # transforms.ColorJitter(brightness= 0.4, contrast= 0.4, saturation= 0.4, hue= 0.1),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_val['res'] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        self.data_dir = data_dir
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']

        self.train_dataset, self.val_dataset = get_cifar100(config['data_loader']['args']['data_dir'], cfg_trainer,
                                                            train=training,
                                                            transform_train=transform_train[trans_type],
                                                            transform_val=transform_val[trans_type])

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset=self.val_dataset)

    def run_loader(self, batch_size, shuffle, validation_split, num_workers, pin_memory):
        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset=self.val_dataset)
