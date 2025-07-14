# Adapted from https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/loaders/cityscapes_loader.py
# Preprocess the cityscapes dataset

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from glob import glob
from tqdm import tqdm

import torch
from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as TF

# import cityscapes info
from supervised_experiments.loaders.cityscapes_labels import (
    city_labels,
    city_ignore_index,
    city_trainID_no_instance,
    city_trainId2label,
    city_categorical_trainId2label,
)


class RandomRotate:
    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, img, label_19, label, disparity, instance, mask):
        angle = random.uniform(-self.max_angle, self.max_angle)

        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
        label_19 = TF.rotate(label_19, angle, interpolation=TF.InterpolationMode.NEAREST)
        label = TF.rotate(label, angle, interpolation=TF.InterpolationMode.NEAREST)
        disparity = TF.rotate(disparity, angle, interpolation=TF.InterpolationMode.NEAREST)
        instance = TF.rotate(instance, angle, interpolation=TF.InterpolationMode.NEAREST)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        return img, label_19, label, disparity, instance, mask


class CityPreprocess(data.Dataset):
    """cityscapesLoader
    https://www.cityscapes-dataset.com
    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    def __init__(self, root, tasks, mode="train", augmentation=True, img_size=(512, 256), n_classes=19):
        """__init__
        :param root:
        :param split:
        :param img_size:
        """
        self.root = os.path.expanduser(root)
        self.tasks = tasks
        self.split = mode
        self.preprocess_folder = "city_preprocess_" + str(img_size[0]) + "_" + str(img_size[1])
        self.augmentation = augmentation
        self.n_classes = n_classes
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        self.disp_factor = 2048.0 / self.img_size[0]

        # pytorch imagenet parameters https://pytorch.org/vision/0.8/models.html - global transformation
        self.img_transorm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # cityscapes: mean = [0.28689553, 0.32513301, 0.28389176] | std = [0.04646142, 0.04200337, 0.04062203]

        if not self._check_city_exists():
            os.makedirs(f"{self.root}/{self.preprocess_folder}", exist_ok=True)
            self.preprocess()

        # preprocessed files
        self.pre_img_files = sorted(glob(os.path.join(self.root, self.preprocess_folder, "image", self.split, "*.npy")))

        self.pre_images_base = os.path.join(self.root, self.preprocess_folder, "image", self.split)
        self.pre_semantic_base = os.path.join(self.root, self.preprocess_folder, "semantic", self.split)
        self.pre_instance_base = os.path.join(self.root, self.preprocess_folder, "instance", self.split)
        self.pre_disparity_base = os.path.join(self.root, self.preprocess_folder, "disparity", self.split)

    def _check_city_exists(self):
        return (
            os.path.exists(os.path.join(self.root, self.preprocess_folder, "image", self.split))
            and os.path.exists(os.path.join(self.root, self.preprocess_folder, "semantic", self.split))
            and os.path.exists(os.path.join(self.root, self.preprocess_folder, "instance", self.split))
            and os.path.exists(os.path.join(self.root, self.preprocess_folder, "disparity", self.split))
        )

    def create_directories(self, split):
        pre_img_root = os.path.join(self.root, self.preprocess_folder, "image", split)
        pre_semantic_root = os.path.join(self.root, self.preprocess_folder, "semantic", split)
        pre_instance_root = os.path.join(self.root, self.preprocess_folder, "instance", split)
        pre_disparity_root = os.path.join(self.root, self.preprocess_folder, "disparity", split)

        # creates directories
        os.makedirs(pre_img_root, exist_ok=True)
        os.makedirs(pre_semantic_root, exist_ok=True)
        os.makedirs(pre_instance_root, exist_ok=True)
        os.makedirs(pre_disparity_root, exist_ok=True)

        return pre_img_root, pre_semantic_root, pre_instance_root, pre_disparity_root

    def preprocess_split(self, pre_img_root, pre_semantic_root, pre_instance_root, pre_disparity_root, files):
        for i, img_path in tqdm(enumerate(files)):
            lbl_path = os.path.join(
                self.annotations_base,
                img_path.split(os.sep)[-3],
                img_path.split(os.sep)[-2],
                os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
            )

            instance_path = os.path.join(
                self.annotations_base,
                img_path.split(os.sep)[-3],
                img_path.split(os.sep)[-2],
                os.path.basename(img_path)[:-15] + "gtFine_instanceIds.png",
            )

            disparity_path = os.path.join(
                self.disparity_base,
                img_path.split(os.sep)[-3],
                img_path.split(os.sep)[-2],
                os.path.basename(img_path)[:-15] + "disparity.png",
            )

            img = np.array(
                Image.open(img_path).resize(self.img_size, resample=Image.Resampling.BILINEAR), dtype=np.float16
            )
            lbl = np.array(
                Image.open(lbl_path).resize(self.img_size, resample=Image.Resampling.NEAREST), dtype=np.int16
            )  # allows ignore index to be negative
            ins = np.array(
                Image.open(instance_path).resize(self.img_size, resample=Image.Resampling.NEAREST), dtype=np.int16
            )

            # reading disparity as recommended here https://github.com/mcordts/cityscapesScripts/blob/master/README.md#dataset-structure
            # disparity = np.array(Image.open(disparity_path).resize(self.img_size, resample=Image.Resampling.NEAREST), dtype=np.float32)
            disparity = cv2.imread(disparity_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            disparity = cv2.resize(disparity, self.img_size, interpolation=cv2.INTER_NEAREST)

            # scales disparity according to resize
            mask = disparity > 0
            disparity[mask] = (disparity[mask] - 1) / (256.0 * self.disp_factor)

            # print("--------reading--------")
            # print("img shape: ", img.shape)
            # print("semantic shape: ", lbl.shape)
            # print("instance shape: ", ins.shape)
            # print("disparity shape: ", disparity.shape)

            img = np.asarray(self.img_transorm(img / 255.0), dtype=np.float16)
            lbl_enc = self.encode_segmap(lbl)
            # ins_enc = self.encode_instancemap(lbl_enc, ins)

            # ins_enc = np.transpose(ins_enc, (2, 0, 1))  # training shape CxHxW

            pre_img_path = os.path.join(pre_img_root, str(i) + ".npy")
            pre_semantic_path = os.path.join(pre_semantic_root, str(i) + ".npy")
            pre_instance_path = os.path.join(pre_instance_root, str(i) + ".npy")
            pre_disparity_path = os.path.join(pre_disparity_root, str(i) + ".npy")

            # 16 bits data when possible to reduce disk memory consumption
            np.save(pre_img_path, img.astype(np.float16))
            np.save(pre_semantic_path, lbl_enc.astype(np.int16))
            np.save(pre_instance_path, ins)
            np.save(pre_disparity_path, disparity.astype(np.float32))

    def save_paths_to_txt(self, filename, path_list_w_root):
        path_list = [path.replace(self.root, "") for path in path_list_w_root]

        with open(filename, "w") as txt_file:
            txt_file.write("\n".join(path_list))

    def load_paths_from_txt(self, filename):
        with open(filename, "r") as txt_file:
            paths = txt_file.readlines()

        paths = [os.path.join(self.root, path.strip()) for path in paths]

        return paths

    def generate_split(self):
        train_file_list_path = f"{self.root}{self.preprocess_folder}/train.txt"
        val_file_list_path = f"{self.root}{self.preprocess_folder}/val.txt"
        test_file_list_path = f"{self.root}{self.preprocess_folder}/test.txt"

        self.images_base = os.path.join(self.root, "city_ori/leftImg8bit_trainvaltest/leftImg8bit")
        self.annotations_base = os.path.join(self.root, "city_ori/gtFine_trainvaltest/gtFine")
        self.disparity_base = os.path.join(self.root, "city_ori/disparity_trainvaltest/disparity")

        print("images path: ", self.images_base)
        print("annotations path: ", self.annotations_base)
        print("disparity path: ", self.disparity_base)

        # split was created already
        if (
            os.path.exists(train_file_list_path)
            and os.path.exists(val_file_list_path)
            and os.path.exists(test_file_list_path)
        ):
            print("reading dataset split from files")
            print(f"train split: {train_file_list_path}")
            print(f"val split: {val_file_list_path}")
            print(f"test split: {test_file_list_path}")

            train_pre_files = self.load_paths_from_txt(train_file_list_path)
            val_pre_files = self.load_paths_from_txt(val_file_list_path)
            test_pre_files = self.load_paths_from_txt(test_file_list_path)

            return train_pre_files, val_pre_files, test_pre_files

        # no split file available
        print("generating dataset split")

        # setting seeds for reproduciability - should run only once on the start,
        # otherwise it could affect the reproduciability.
        np.random.seed(0)
        random.seed(0)

        # original splits
        train_files = sorted(glob(self.images_base + "/train/*/*.png"))
        val_files = sorted(glob(self.images_base + "/val/*/*.png"))

        # process files split
        train_pre_files = sorted(np.random.choice(train_files, int(0.8 * len(train_files)), replace=False))
        val_pre_files = list(
            filter(lambda i: i not in train_pre_files, train_files)
        )  # removes files previously choosen

        test_pre_files = val_files

        self.save_paths_to_txt(train_file_list_path, train_pre_files)
        self.save_paths_to_txt(val_file_list_path, val_pre_files)
        self.save_paths_to_txt(test_file_list_path, test_pre_files)

        return train_pre_files, val_pre_files, test_pre_files

    def preprocess(self):
        print("preprocessing cityscapes dataset")

        train_pre_files, val_pre_files, test_pre_files = self.generate_split()

        # proprocess all splits
        print("train split size:", len(train_pre_files))
        pre_img_root, pre_semantic_root, pre_instance_root, pre_disparity_root = self.create_directories("train")
        self.preprocess_split(pre_img_root, pre_semantic_root, pre_instance_root, pre_disparity_root, train_pre_files)

        print("validation split size:", len(val_pre_files))
        pre_img_root, pre_semantic_root, pre_instance_root, pre_disparity_root = self.create_directories("val")
        self.preprocess_split(pre_img_root, pre_semantic_root, pre_instance_root, pre_disparity_root, val_pre_files)

        print("test split size:", len(test_pre_files))
        pre_img_root, pre_semantic_root, pre_instance_root, pre_disparity_root = self.create_directories("test")
        self.preprocess_split(pre_img_root, pre_semantic_root, pre_instance_root, pre_disparity_root, test_pre_files)

    def __len__(self):
        return len(self.pre_img_files)

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        # image paths
        img_path = self.pre_img_files[index]
        semantic_path = os.path.join(self.pre_semantic_base, os.path.basename(img_path)[:-4] + ".npy")
        instance_path = os.path.join(self.pre_instance_base, os.path.basename(img_path)[:-4] + ".npy")
        disparity_path = os.path.join(self.pre_disparity_base, os.path.basename(img_path)[:-4] + ".npy")

        # load preprocessed image files
        img = np.load(img_path)
        semantic = np.load(semantic_path)
        instance = np.load(instance_path)
        disparity = np.load(disparity_path)

        semantic_19 = np.copy(semantic)
        if self.n_classes == 7:
            semantic = self.encode_7_segmap(semantic)

        img = torch.as_tensor(img, dtype=torch.float)
        semantic = torch.as_tensor(semantic, dtype=torch.float).unsqueeze(0)
        semantic_19 = torch.as_tensor(semantic_19, dtype=torch.float).unsqueeze(0)
        instance = torch.as_tensor(instance, dtype=torch.float).unsqueeze(0)
        disparity = torch.as_tensor(disparity, dtype=torch.float).unsqueeze(0)

        # apply data augmentation if required
        if self.augmentation:
            mask = torch.ones_like(disparity)
            img, semantic_19, semantic, disparity, instance, mask = RandomRotate(10)(
                img, semantic_19, semantic, disparity, instance, mask
            )

            # print("aug img shape: ", img.shape)
            # print("aug semantic shape: ", semantic.shape)
            # print("aug instance shape: ", instance.shape)
            # print("aug disparity shape: ", disparity.shape)
            # print("aug mask shape: ", mask.shape)

            semantic[mask == 0] = city_ignore_index
            semantic_19[mask == 0] = city_ignore_index

            if torch.rand(1) < 0.5:
                img = torch.flip(img, dims=[2])
                semantic_19 = torch.flip(semantic_19, dims=[2])
                semantic = torch.flip(semantic, dims=[2])
                instance = torch.flip(instance, dims=[2])
                disparity = torch.flip(disparity, dims=[2])

        semantic_19 = semantic_19.squeeze()
        semantic = semantic.squeeze()
        instance = instance.squeeze()

        ins_enc = self.encode_instancemap(semantic_19, instance)
        ins_enc = torch.as_tensor(np.transpose(ins_enc, (2, 0, 1)), dtype=torch.float)

        data = [img]

        for t in self.tasks:
            if t == "S":
                data.append(semantic)
            elif t == "D":
                data.append(disparity)
            elif t == "I":
                data.append(ins_enc)
            else:
                raise NotImplementedError(f"task {t} not implemented in cityscapes")

        return data

    # ----------------------------------------------------------------
    # ----------------------ENCONDING FUNCTIONS-----------------------
    # ----------------------------------------------------------------

    # semantic segmentation
    def encode_segmap(self, sem_label):
        temp = np.copy(sem_label)
        for label in city_labels:
            sem_label[temp == label.id] = label.trainId

        return sem_label

    # 7 classes enconding - reads from encoded semantic label
    def encode_7_segmap(self, sem_label):
        temp = np.copy(sem_label)
        for label in city_labels:
            sem_label[temp == label.trainId] = label.categoryTrainId

        return sem_label

    def encode_instancemap(self, sem_label, instance):
        # sem label should be encoded as trainID
        instance[sem_label == city_ignore_index] = city_ignore_index

        for no_ins_id in city_trainID_no_instance:
            instance[sem_label == no_ins_id] = city_ignore_index

        instance[instance == 0] = city_ignore_index

        instance_ids = np.unique(instance)

        sh = instance.shape
        ymap, xmap = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]), indexing="ij")

        ins_out = np.ones((sh[0], sh[1], 2)) * city_ignore_index

        for ins_id in instance_ids:
            if ins_id == city_ignore_index:
                continue

            ins_mask = instance == ins_id

            mean_y, mean_x = np.mean(ymap[ins_mask]), np.mean(xmap[ins_mask])

            ins_out[ins_mask, 0] = ymap[ins_mask] - mean_y
            ins_out[ins_mask, 1] = xmap[ins_mask] - mean_x

        return ins_out

    # ----------------------------------------------------------------
    # -----------------------DECODING FUNCTIONS-----------------------
    # ----------------------------------------------------------------

    # semantic segmentation
    def decode_segmap(self, temp):
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))

        if self.n_classes == 7:
            for l in range(0, self.n_classes):
                rgb[temp == l] = city_categorical_trainId2label[l].color
        else:
            for l in range(0, self.n_classes):
                rgb[temp == l] = city_trainId2label[l].color

        rgb /= 255.0

        return rgb

    # instance segmentation
    def decode_instance(self, temp):
        instance_center_dist = np.linalg.norm(temp, axis=0)

        return instance_center_dist


if __name__ == "__main__":
    # preprocess images for faster loading
    with open("supervised_experiments/configs.json") as config_params:
        configs = json.load(config_params)

    local_path = configs["cityscapes"]["path"]
    random_seed = 0
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    g = torch.Generator()
    g.manual_seed(random_seed)

    # Creates the preprocessed version of the dataset
    dataset = CityPreprocess(local_path, tasks=["S", "I", "D"], mode="train", img_size=(256, 128), augmentation=False)
    dataset = CityPreprocess(local_path, tasks=["S", "I", "D"], mode="train", augmentation=False)

    os.makedirs("plots/cityscapes/debug/", exist_ok=True)

    batch_size = 1
    trainloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
    for i, batch_data in enumerate(trainloader):
        imgs, labels, instances, disparity = batch_data
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        img_show = (imgs[0] - torch.min(imgs[0])) / (torch.max(imgs[0]) - torch.min(imgs[0]))

        plt.figure()
        plt.axis(False)
        plt.imshow(img_show)
        plt.tight_layout()
        plt.savefig("plots/cityscapes/debug/img.png", bbox_inches="tight", pad_inches=0, dpi=1200)
        plt.close()

        plt.figure()
        plt.axis(False)
        plt.imshow(dataset.decode_segmap(labels.numpy()[0]))
        plt.tight_layout()
        plt.savefig("plots/cityscapes/debug/sem.png", bbox_inches="tight", pad_inches=0, dpi=1200)
        plt.close()

        plt.figure()
        plt.axis(False)
        plt.imshow(disparity[0][0])
        plt.tight_layout()
        plt.savefig("plots/cityscapes/debug/disp.png", bbox_inches="tight", pad_inches=0, dpi=1200)
        plt.close()

        plt.figure()
        plt.axis(False)
        plt.imshow(instances[0][1])
        plt.tight_layout()
        plt.savefig("plots/cityscapes/debug/inst_X.png", bbox_inches="tight", pad_inches=0, dpi=1200)
        plt.close()

        plt.figure()
        plt.axis(False)
        plt.imshow(instances[0][0])
        plt.tight_layout()
        plt.savefig("plots/cityscapes/debug/inst_Y.png", bbox_inches="tight", pad_inches=0, dpi=1200)
        plt.close()

        break
