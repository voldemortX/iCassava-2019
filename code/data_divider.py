import os
import shutil
import random


# Generate dev set from training set(which is structured for torchvision.datasets.ImageFolder)
# xx% from each category
def divide_data(base_train, base_dev, dev_percentage):
    # Find sub paths(categories)
    categories = []
    for i in os.listdir(base_train):
        categories.append(i)

    # For each category
    random.seed(1)  # Seed needs to be fixed to be reproducible
    for category in categories:
        pics = []
        for i in os.listdir(os.path.join(base_train, category)):
            pics.append(i)

        random.shuffle(pics)
        dev_set = pics[0: int(len(pics) * dev_percentage)]
        os.mkdir(os.path.join(base_dev, category))
        for i in dev_set:
            shutil.move(os.path.join(base_train, category, i), os.path.join(base_dev, category, i))


if __name__ == "__main__":
    train_dirname = "../data/train"
    dev_dirname = "../data/dev"
    dev_ratio = 0.2
    print("Dividing...")
    divide_data(base_train=train_dirname, base_dev=dev_dirname, dev_percentage=dev_ratio)
    print("Done!")
