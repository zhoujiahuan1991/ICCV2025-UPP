from datasets import data_transforms
from torchvision import transforms
train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScaleAndTranslate(),
        # data_transforms.PointcloudScaleAndTranslate(scale_low=0.9, scale_high=1.1, translate_range=0),
        data_transforms.PointcloudRotate(),
    ]
)

rotate = transforms.Compose(
    [   
        data_transforms.PointcloudRotate()
    ]
)

scale_translate = transforms.Compose(
    [   
        data_transforms.PointcloudScaleAndTranslate()
    ]
)

jitter = transforms.Compose(
    [   
        # data_transforms.PointcloudJitter(std=0.05, clip=0.1)
        data_transforms.PointcloudJitter(std=0.01, clip=0.03)
    ]
)

add_noise = transforms.Compose(
    [   
        data_transforms.AddNoise(noise_std_min=0.005, noise_std_max=0.03)
    ]
)

normalize = transforms.Compose(
    [   
        data_transforms.NormalizeUnitSphere()
    ]
)

test_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)