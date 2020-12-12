import os
import shutil

import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv("train_public.csv")
    for i in [0, 1]:
        os.makedirs(f"dataset/{i}", exist_ok=True)
        names = data[data["disease_flag"] == i]["name"]
        for name in names:
            shutil.copyfile(f"train_public/{name}", f"dataset/{i}/{name}")
