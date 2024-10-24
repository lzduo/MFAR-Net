import os
import numpy as np
import matplotlib.pyplot as plt


def draw(data_dict, path, args, adds=None):
    for k, v in data_dict.items():
        plt.plot(v)
        plt.xticks(np.arange(0, args.epochs + 1, 10))
        if not ("loss" in str(k) or "lr" in str(k)):
            plt.yticks(np.arange(0, 1, 0.1))
        plt.xlabel('epoch')
        plt.ylabel(str(k))
        if isinstance(adds, str):
            plt.savefig(os.path.join(path, f"{k}_{adds}.png"))
        else:
            plt.savefig(os.path.join(path, f"{k}.png"))
        plt.clf()


def draw_compare(train_dict, val_dict, path, args, adds=None):
    for k1, v1 in train_dict.items():
        for k2, v2 in val_dict.items():
            if k1 == k2:
                plt.plot(v1, label='Train Loss')
                plt.plot(v2, label='Val Loss')
                plt.xticks(np.arange(0, args.epochs + 1, 10))
                plt.xlabel('epoch')
                plt.ylabel(str(k1))

                if isinstance(adds, str):
                    plt.savefig(os.path.join(path, f"{k1}_{adds}.png"))
                else:
                    plt.savefig(os.path.join(path, f"{k1}.png"))
                plt.clf()
            else:
                continue


def save_evaluations(data_dict, path):
    nums = len(list(data_dict.values())[0])
    with open(os.path.join(path, "evaluations.txt"), "w") as f:
        for i in range(nums):
            f.writelines(f"----------------epoch:{i}----------------\n")
            for k, v in data_dict.items():
                f.writelines(f"{k}:{v[i]}\n")
            f.writelines(f"\n")
    f.close()


def save_parameters(args, path):
    with open(os.path.join(path, "parameters.txt"), "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    f.close()
