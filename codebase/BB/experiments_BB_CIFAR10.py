import os
import random
from collections import OrderedDict
from datetime import datetime

import clip
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from tqdm import tqdm

import utils
from BB.models.BB_ResNet import ResNet
from BB.models.Clip_classifier import Classifier
from Logger.logger_cubs import Logger_CUBS
from dataset.dataset_mnist import Dataset_mnist
from dataset.utils_dataset import get_dataset_with_image_and_attributes


def train_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device, download_root=args.checkpoints)

    # Load the dataset
    if args.dataset == "mnist":
        train_set, train_attributes = get_dataset_with_image_and_attributes(
            data_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/data/MNIST_EVEN_ODD",
            json_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/scripts_data",
            dataset_name="mnist",
            mode="train",
            attribute_file="attributes.npy"
        )

        val, val_attributes = get_dataset_with_image_and_attributes(
            data_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/data/MNIST_EVEN_ODD",
            json_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/scripts_data",
            dataset_name="mnist",
            mode="val",
            attribute_file="attributes.npy"
        )

        test_set, test_attributes = get_dataset_with_image_and_attributes(
            data_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/data/MNIST_EVEN_ODD",
            json_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/scripts_data",
            dataset_name="mnist",
            mode="test",
            attribute_file="attributes.npy"
        )

        # train_transform = get_transforms(size=224)
        transform = transforms.Compose([
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        dataset = Dataset_mnist(train_set, transform)
        train = DataLoader(dataset, batch_size=10, shuffle=True)
    else:
        train = CIFAR10(args.data_root, download=True, train=True, transform=preprocess)
        test = CIFAR10(args.data_root, download=True, train=False, transform=preprocess)

    def get_features(dataset, mode):
        # feature_path = os.path.join(args.output, f"{mode}_features")
        # os.makedirs(feature_path, exist_ok=True)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                features = model.encode_image(images.to(device))
                all_features.append(features)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

    # Calculate the image features
    train_features, train_labels = get_features(train, mode="train")
    test_features, test_labels = get_features(test, mode="test")

    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")
    print(train.classes)


def train(args):
    device = utils.get_device()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = f"lr_{args.lr}_epochs_{args.epochs}"

    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "BB", root, args.arch)
    output_path = os.path.join(args.output, args.dataset, "BB", root, args.arch)
    tb_logs_path = os.path.join(args.logs, args.dataset, "BB", f"{root}_{args.arch}")
    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)

    device = utils.get_device()
    print(f"Device: {device}")

    if args.dataset == "mnist":
        train_set, train_attributes = get_dataset_with_image_and_attributes(
            data_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/data/MNIST_EVEN_ODD",
            json_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/scripts_data",
            dataset_name="mnist",
            mode="train",
            attribute_file="attributes.npy"
        )

        val_set, val_attributes = get_dataset_with_image_and_attributes(
            data_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/data/MNIST_EVEN_ODD",
            json_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/scripts_data",
            dataset_name="mnist",
            mode="val",
            attribute_file="attributes.npy"
        )

        test_set, test_attributes = get_dataset_with_image_and_attributes(
            data_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/data/MNIST_EVEN_ODD",
            json_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/scripts_data",
            dataset_name="mnist",
            mode="test",
            attribute_file="attributes.npy"
        )

        # train_transform = get_transforms(size=224)
        transform = transforms.Compose([
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        train_dataset = Dataset_mnist(train_set, train_attributes, transform)
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

        test_dataset = Dataset_mnist(test_set, test_attributes, transform)
        val_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

        args.pretrained = True
        classifier = ResNet(
            dataset=args.dataset, pre_trained=args.pretrained, n_class=len(args.labels),
            model_choice=args.arch, layer="layer4"
        ).to(device)
    else:
        train_set = CIFAR10(args.data_root, download=True, train=True, transform=preprocess)
        val_set = CIFAR10(args.data_root, download=True, train=False, transform=preprocess)

        train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=True)


        clip_model, preprocess = clip.load("RN50", device, download_root=chk_pt_path)
        classifier = Classifier(in_features=1024, out_features=len(args.labels)).to(device)

    solver = torch.optim.SGD(
        classifier.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()
    final_parameters = OrderedDict(
        arch=[args.arch],
        dataset=[args.dataset],
        now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')]
    )

    run_id = utils.get_runs(final_parameters)[0]
    run_manager = Logger_CUBS(1, chk_pt_path, tb_logs_path, output_path, train_loader, val_loader, len(args.labels))
    fit(
        args.epochs,
        classifier,
        criterion,
        solver,
        train_loader,
        val_loader,
        run_manager,
        args.dataset,
        run_id,
        device
    )


def fit(
        epochs,
        classifier,
        criterion,
        solver,
        train_loader,
        val_loader,
        run_manager,
        dataset,
        run_id,
        device
):
    run_manager.begin_run(run_id)
    for epoch in range(epochs):
        run_manager.begin_epoch()
        classifier.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, (images, labels, _) in enumerate(train_loader):
                solver.zero_grad()
                images = images.to(device)
                labels = labels.to(device)
                y_hat = classifier(images)
                # print(y_hat)
                train_loss = criterion(y_hat, labels)
                train_loss.backward()
                solver.step()

                run_manager.track_train_loss(train_loss.item())
                run_manager.track_total_train_correct_per_epoch(y_hat, labels)
                t.set_postfix(
                    epoch='{0}'.format(epoch),
                    training_loss='{:05.3f}'.format(run_manager.epoch_train_loss))
                t.update()

        classifier.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, (images, labels, _) in enumerate(val_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    y_hat = classifier(images)
                    val_loss = criterion(y_hat, labels)

                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_per_epoch(y_hat, labels)
                    t.set_postfix(
                        epoch='{0}'.format(epoch),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss))
                    t.update()

        run_manager.end_epoch(classifier)
        print(f"Epoch: [{epoch + 1}/{epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Train_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} (%) "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"Best_Val_Accuracy: {round(run_manager.get_final_best_val_accuracy(), 4)} (%)  "
              f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)}")

    run_manager.end_run()
