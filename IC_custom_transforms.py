import random
from PIL import Image
from torchvision.transforms import ColorJitter

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img = sample['img']
        if random.random() < self.p:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)  # Fixed
        return {'img': img}

class RandomColorJitter(object):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1,p=0.5):
        self.jitter = ColorJitter(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, sample):
        img = sample['img']
        if random.random() < self.p:
            img = self.jitter(img)
        return {'img': img}

def transform(example_batch, processor, is_training=True):
    # Apply augmentations only during training
    transforms = [
        RandomHorizontalFlip(p=0.5),
        RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ] if is_training else []

    # Process each image in the batch
    processed_images = []
    for img in example_batch["img"]:
        sample = {'img': img.convert("RGB")}
        # Apply augmentations
        for t in transforms:
            sample = t(sample)
        processed_images.append(sample['img'])

    # Use processor to convert images to tensors and normalize
    inputs = processor(processed_images, return_tensors="pt")
    inputs["labels"] = torch.tensor(example_batch["label"])
    return inputs

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }