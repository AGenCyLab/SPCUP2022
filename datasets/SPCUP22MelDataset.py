from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

RESIZE = 224


class SPCUP22MelDataset(Dataset):
    def __init__(self, annotations_df, *, mode="train"):
        self.annotations_df = annotations_df
        self.mode = mode

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, index):
        target = -1

        if self.mode == "eval":
            path_as_label = self.annotations_df.iloc[index, 1]
            image_path = path_as_label.replace("wav", "jpg")
        else:
            target = self.annotations_df.iloc[index, 1]
            image_path = self.annotations_df.iloc[index, 0].replace(
                "wav", "jpg"
            )

        image_bytes = Image.open(image_path).convert("L")
        w, h = image_bytes.size

        image = np.array(image_bytes)

        transforms = A.Compose(
            [
                A.Resize(height=RESIZE, width=RESIZE),
                ToTensorV2(),
            ],
        )

        image = transforms(image=image)["image"]

        return image, target, image_path
