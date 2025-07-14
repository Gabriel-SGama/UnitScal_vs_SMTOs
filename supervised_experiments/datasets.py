# Code adapted from https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/

import torch
import torch_geometric
from torchvision import transforms
from supervised_experiments.loaders.multi_mnist_loader import MNIST
from supervised_experiments.loaders.city_preprocess import CityPreprocess
from supervised_experiments.loaders.QM9 import QM9_dataset


def mnist_transformer():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


def get_dataset(args, configs, tasks, generator=None, worker_init_fn=None, train=True):
    dataset = args.dataset
    batch_size = args.batch_size

    if "mnist" in dataset:
        if train:
            train_dst = MNIST(
                root=configs["mnist"]["path"],
                tasks=tasks,
                split="train",
                download=True,
                transform=mnist_transformer(),
            )
            train_loader = torch.utils.data.DataLoader(
                train_dst,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                generator=generator,
                worker_init_fn=worker_init_fn,
            )

            val_dst = MNIST(
                root=configs["mnist"]["path"],
                tasks=tasks,
                split="val",
                download=True,
                transform=mnist_transformer(),
            )
            val_loader = torch.utils.data.DataLoader(
                val_dst,
                batch_size=100,
                shuffle=True,
                num_workers=2,
                generator=generator,
                worker_init_fn=worker_init_fn,
            )
            return train_loader, val_loader
        else:
            test_dst = MNIST(
                root=configs["mnist"]["path"],
                tasks=tasks,
                split="test",
                download=True,
                transform=mnist_transformer(),
            )
            test_loader = torch.utils.data.DataLoader(
                test_dst,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2,
                generator=generator,
                worker_init_fn=worker_init_fn,
            )
            return test_loader, test_dst

    if "city" in dataset:
        if train:
            train_loader = torch.utils.data.DataLoader(
                dataset=CityPreprocess(
                    root=configs["cityscapes"]["path"],
                    tasks=tasks,
                    mode="train",
                    img_size=tuple(args.shape),
                    augmentation=args.aug,
                    n_classes=args.n_classes,
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                generator=generator,
                worker_init_fn=worker_init_fn,
            )

            val_loader = torch.utils.data.DataLoader(
                dataset=CityPreprocess(
                    root=configs["cityscapes"]["path"],
                    tasks=tasks,
                    mode="val",
                    img_size=tuple(args.shape),
                    augmentation=False,
                    n_classes=args.n_classes,
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                generator=generator,
                worker_init_fn=worker_init_fn,
            )

            return train_loader, val_loader

        else:
            test_dst = CityPreprocess(
                root=configs["cityscapes"]["path"],
                tasks=tasks,
                mode="test",
                img_size=tuple(args.shape),
                augmentation=False,
                n_classes=args.n_classes,
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dst,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                generator=generator,
                worker_init_fn=worker_init_fn,
            )

            return test_loader, test_dst

    if "QM9" in dataset:
        if train:
            train_dataset = QM9_dataset(configs["QM9"]["path"], tasks, split="train")
            train_loader = torch_geometric.loader.DataLoader(
                train_dataset.selected_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                generator=generator,
                worker_init_fn=worker_init_fn,
            )

            val_dataset = QM9_dataset(configs["QM9"]["path"], tasks, split="val")
            val_loader = torch_geometric.loader.DataLoader(
                val_dataset.selected_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                generator=generator,
                worker_init_fn=worker_init_fn,
            )

            return train_loader, val_loader

        else:
            test_dst = QM9_dataset(configs["QM9"]["path"], tasks, split="test")
            test_loader = torch_geometric.loader.DataLoader(
                test_dst.selected_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                generator=generator,
                worker_init_fn=worker_init_fn,
            )

            return test_loader, test_dst
