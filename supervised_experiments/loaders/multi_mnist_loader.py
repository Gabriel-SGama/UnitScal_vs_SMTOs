# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py

import os
import os.path
import errno
import numpy as np
import codecs
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms


class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, tasks, split="train", transform=None, download=False):
        assert split in ["train", "val", "test"]

        self.urls = [
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
        ]
        self.raw_folder = "raw"

        self.processed_folder = "processed"
        self.training_file = "training.pt"
        self.test_file = "test.pt"
        self.multi_training_file = "multi_training.pt"
        self.multi_validation_file = "multi_validation.pt"
        self.multi_test_file = "multi_test.pt"

        self.root = os.path.expanduser(root)

        self.transform = transform
        self.split = split
        self.tasks = tasks

        if download:
            self.download()

        if not self._check_multi_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download MNIST and generate a random MultiMNIST"
            )

        if self.split == "train":
            (
                self.train_left_data,
                self.train_right_data,
                self.train_data,
                self.train_labels_l,
                self.train_labels_r,
            ) = torch.load(os.path.join(self.root, self.processed_folder, self.multi_training_file))
        elif self.split == "val":
            (
                self.validation_left_data,
                self.validation_right_data,
                self.validation_data,
                self.validation_labels_l,
                self.validation_labels_r,
            ) = torch.load(os.path.join(self.root, self.processed_folder, self.multi_validation_file))
        else:
            (
                self.test_left_data,
                self.test_right_data,
                self.test_data,
                self.test_labels_l,
                self.test_labels_r,
            ) = torch.load(os.path.join(self.root, self.processed_folder, self.multi_test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            list: (image, target1, ..., targetN) where N is index of the target to each task.
        """
        if self.split == "train":
            img_L, img_R, img, target_l, target_r = (
                self.train_left_data[index],
                self.train_right_data[index],
                self.train_data[index],
                self.train_labels_l[index],
                self.train_labels_r[index],
            )
        elif self.split == "val":
            img_L, img_R, img, target_l, target_r = (
                self.validation_left_data[index],
                self.validation_right_data[index],
                self.validation_data[index],
                self.validation_labels_l[index],
                self.validation_labels_r[index],
            )
        else:
            img_L, img_R, img, target_l, target_r = (
                self.test_left_data[index],
                self.test_right_data[index],
                self.test_data[index],
                self.test_labels_l[index],
                self.test_labels_r[index],
            )

        if "RL" in self.tasks:
            img_L = Image.fromarray(img_L.numpy().astype(np.uint8), mode="L")
        if "RR" in self.tasks:
            img_R = Image.fromarray(img_R.numpy().astype(np.uint8), mode="L")

        img = Image.fromarray(img.numpy().astype(np.uint8), mode="L")

        if self.transform is not None:
            if "RL" in self.tasks:
                img_L = self.transform(img_L).squeeze()
            if "RR" in self.tasks:
                img_R = self.transform(img_R).squeeze()
            img = self.transform(img)

        data = []
        data.append(img)
        for task in self.tasks:
            if task == "CL":
                data.append(target_l)
            elif task == "CR":
                data.append(target_r)
            elif task == "RL":
                data.append(img_L)
            elif task == "RR":
                data.append(img_R)
            else:
                raise NotImplementedError(f"task {task} not supported")

        return data

    def __len__(self):
        if self.split == "train":
            return len(self.train_data)
        elif self.split == "val":
            return len(self.validation_data)
        else:
            return len(self.test_data)

    def _check_multi_exists(self):
        return (
            os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_training_file))
            and os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_test_file))
            and os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_validation_file))
        )

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_multi_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder), exist_ok=True)
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print("Downloading " + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition("/")[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, "wb") as f:
                f.write(data.read())
            with open(file_path.replace(".gz", ""), "wb") as out_f, gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print("Processing...")
        # Create train-set images from MNIST's original training set.
        using_previsous_indexes = False
        if (
            os.path.exists(os.path.join(self.root, self.processed_folder, "train_right_idx.txt"))
            and os.path.exists(os.path.join(self.root, self.processed_folder, "val_right_idx.txt"))
            and os.path.exists(os.path.join(self.root, self.processed_folder, "test_right_idx.txt"))
        ):

            print("Using previously saved indexes...")
            train_right_idx = np.loadtxt(
                os.path.join(self.root, self.processed_folder, "train_right_idx.txt"),
                dtype=int,
            )
            val_right_idx = np.loadtxt(
                os.path.join(self.root, self.processed_folder, "val_right_idx.txt"),
                dtype=int,
            )
            test_right_idx = np.loadtxt(
                os.path.join(self.root, self.processed_folder, "test_right_idx.txt"),
                dtype=int,
            )
            using_previsous_indexes = True
        else:
            train_right_idx = None
            val_right_idx = None
            test_right_idx = None

        (
            mnist_ims,
            left_imgs,
            right_imgs,
            multi_mnist_ims,
            train_left_idx,
            train_right_idx,
        ) = create_multimnist_images(
            os.path.join(self.root, self.raw_folder, "train-images-idx3-ubyte"),
            split="train",
            right_indices=train_right_idx,
        )
        # Create validation set images from MNIST's original training set.
        (
            vmnist_ims,
            vleft_imgs,
            vright_imgs,
            vmulti_mnist_ims,
            val_left_idx,
            val_right_idx,
        ) = create_multimnist_images(
            os.path.join(self.root, self.raw_folder, "train-images-idx3-ubyte"),
            split="val",
            right_indices=val_right_idx,
        )
        # Create test set images from MNIST's original test set (the second image to be overlapped is randomly chosen).
        (
            tmnist_ims,
            tleft_imgs,
            tright_imgs,
            tmulti_mnist_ims,
            test_left_idx,
            test_right_idx,
        ) = create_multimnist_images(
            os.path.join(self.root, self.raw_folder, "t10k-images-idx3-ubyte"),
            split="test",
            right_indices=test_right_idx,
        )

        mnist_labels, multi_mnist_labels_l, multi_mnist_labels_r = create_multimnist_labels(
            os.path.join(self.root, self.raw_folder, "train-labels-idx1-ubyte"),
            train_left_idx,
            train_right_idx,
        )
        vmnist_labels, vmulti_mnist_labels_l, vmulti_mnist_labels_r = create_multimnist_labels(
            os.path.join(self.root, self.raw_folder, "train-labels-idx1-ubyte"),
            val_left_idx,
            val_right_idx,
        )
        tmnist_labels, tmulti_mnist_labels_l, tmulti_mnist_labels_r = create_multimnist_labels(
            os.path.join(self.root, self.raw_folder, "t10k-labels-idx1-ubyte"),
            test_left_idx,
            test_right_idx,
        )

        if not using_previsous_indexes:
            print("Saving indexes...")
            with open(
                os.path.join(self.root, self.processed_folder, "train_right_idx.txt"),
                "w",
            ) as f:
                for item in train_right_idx:
                    f.write("%s\n" % item)
            with open(os.path.join(self.root, self.processed_folder, "val_right_idx.txt"), "w") as f:
                for item in val_right_idx:
                    f.write("%s\n" % item)
            with open(
                os.path.join(self.root, self.processed_folder, "test_right_idx.txt"),
                "w",
            ) as f:
                for item in test_right_idx:
                    f.write("%s\n" % item)

        multi_mnist_training_set = (
            left_imgs,
            right_imgs,
            multi_mnist_ims,
            multi_mnist_labels_l,
            multi_mnist_labels_r,
        )
        multi_mnist_validation_set = (
            vleft_imgs,
            vright_imgs,
            vmulti_mnist_ims,
            vmulti_mnist_labels_l,
            vmulti_mnist_labels_r,
        )
        multi_mnist_test_set = (
            tleft_imgs,
            tright_imgs,
            tmulti_mnist_ims,
            tmulti_mnist_labels_l,
            tmulti_mnist_labels_r,
        )

        with open(
            os.path.join(self.root, self.processed_folder, self.multi_training_file),
            "wb",
        ) as f:
            torch.save(multi_mnist_training_set, f)
        with open(
            os.path.join(self.root, self.processed_folder, self.multi_validation_file),
            "wb",
        ) as f:
            torch.save(multi_mnist_validation_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.multi_test_file), "wb") as f:
            torch.save(multi_mnist_test_set, f)
        print("Done!")

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = self.split
        fmt_str += "    Split: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, "hex"), 16)


def create_multimnist_labels(path, left_indices, right_indices):
    with open(path, "rb") as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        nom_length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        length = len(left_indices)
        multi_labels_l = np.zeros(length, dtype=np.int64)
        multi_labels_r = np.zeros(length, dtype=np.int64)
        for im_id in range(len(left_indices)):
            multi_labels_l[im_id] = parsed[left_indices[im_id]]
            multi_labels_r[im_id] = parsed[right_indices[im_id]]
        return (
            torch.from_numpy(parsed.copy()).view(nom_length).long(),
            torch.from_numpy(multi_labels_l.copy()).view(length).long(),
            torch.from_numpy(multi_labels_r.copy()).view(length).long(),
        )


def create_multimnist_images(path, split="train", right_indices=None):
    assert split in ["train", "val", "test"]

    with open(path, "rb") as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        nom_length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        pv = parsed.reshape(nom_length, num_rows, num_cols)

        if split == "train":
            assert nom_length == 60000, "Need to pass the original MNIST training set to create a training split"
            val_size = 10000  # same size as original test split
            start = 0
            end = nom_length - val_size
        elif split == "val":
            assert nom_length == 60000, "Need to pass the original MNIST training set to create a validation split"
            val_size = 10000  # same size as original test split
            start = nom_length - val_size
            end = nom_length
        else:  # split == "test"
            assert nom_length == 10000, "Need to pass the original MNIST test set to create a test split"
            start = 0
            end = nom_length
        length = end - start

        left_data = np.zeros((1 * length, num_rows, num_cols))
        right_data = np.zeros((1 * length, num_rows, num_cols))
        multi_data = np.zeros((1 * length, num_rows, num_cols))

        left_indices = list(range(start, end))
        if right_indices is None:
            right_indices = np.random.randint(low=start, high=end, size=end).tolist()

        for im_id in range(len(left_indices)):
            lim = pv[left_indices[im_id], :, :]
            rim = pv[right_indices[im_id], :, :]

            multi_rim = rim.astype(float)
            multi_rim = np.clip(multi_rim, 0, 255).astype(np.uint8)

            new_im = np.zeros((36, 36))
            new_im[0:28, 0:28] = lim
            new_im[6:34, 6:34] = multi_rim
            new_im[6:28, 6:28] = np.maximum(lim[6:28, 6:28], multi_rim[0:22, 0:22])
            multi_data_im = np.array(Image.fromarray(new_im).resize((28, 28), resample=Image.NEAREST))
            multi_data[im_id, :, :] = multi_data_im

            left_data[im_id, :, :] = pv[left_indices[im_id], :, :]
            right_data[im_id, :, :] = pv[right_indices[im_id], :, :]

        return (
            torch.from_numpy(parsed.copy()).view(nom_length, num_rows, num_cols),
            torch.from_numpy(left_data.copy()).view(length, num_rows, num_cols),
            torch.from_numpy(right_data.copy()).view(length, num_rows, num_cols),
            torch.from_numpy(multi_data.copy()).view(length, num_rows, num_cols),
            left_indices,
            right_indices,
        )


if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = [8, 2.5]
    plt.rcParams["figure.autolayout"] = True

    # ensures same dataset split (can differ between computers though)
    torch.manual_seed(0)
    np.random.seed(0)
    np.random.seed(0)
    g = torch.Generator()
    g.manual_seed(0)

    def global_transformer():
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    os.makedirs("datasets/", exist_ok=True)
    os.makedirs("plots/mnist/", exist_ok=True)

    mnist_dataset = MNIST(
        "datasets/multimnist_dataset/",
        tasks=["CL", "CR", "RL", "RR"],
        split="train",
        download=True,
        transform=global_transformer(),
    )

    mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=1, shuffle=True, num_workers=1)

    for mnist_data in mnist_loader:
        img, target_l, target_r, img_L, img_R = mnist_data

        # saves an image with all cases
        plt.rcParams["figure.figsize"] = [8, 2.5]
        plt.rcParams["figure.autolayout"] = True

        fig, axs = plt.subplots(1, 3)

        axs[0].imshow(img_L.numpy()[0], cmap="gray")
        axs[1].imshow(img_R.numpy()[0], cmap="gray")
        axs[2].imshow(img.numpy()[0][0], cmap="gray")

        axs[0].set_title("Left")
        axs[1].set_title("Right")
        axs[2].set_title("Joint")

        plt.savefig("plots/mnist/debug/multi.png")

        # saves each image separately
        plt.rcParams["figure.figsize"] = [2.5, 2.5]
        plt.rcParams["figure.autolayout"] = True
        plt.figure()
        plt.axis(False)
        plt.imshow(img_L.numpy()[0], cmap="gray")
        plt.savefig("plots/mnist/debug/img_L.png", bbox_inches="tight")
        plt.figure()
        plt.axis(False)
        plt.imshow(img_R.numpy()[0], cmap="gray")
        plt.savefig("plots/mnist/debug/img_R.png", bbox_inches="tight")
        plt.figure()
        plt.axis(False)
        plt.imshow(img.numpy()[0][0], cmap="gray")
        plt.savefig("plots/mnist/debug/img_joint.png", bbox_inches="tight")

        break
