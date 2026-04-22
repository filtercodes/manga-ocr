import albumentations as A
import cv2
from manga_ocr_dev.synthetic_data_generator.crt_emulator import CRTDistortion, GameBoyFilter, SmoothUpscale

def build_dirt_pipeline(mode="crt", gb_palette="random"):
    """
    Constructs the randomized Albumentations pipeline based on the selected hardware mode.
    """
    hardware_transforms = []

    if mode == "crt":
        hardware_transforms.append(
            CRTDistortion(
                k1_range=(-0.04, 0.07),
                k2_range=(-0.01, 0.01),
                bloom_scale_range=(0.02, 0.1),
                scanline_alpha_range=(0.4, 0.7),
                mask_types=("aperture", "slot", "shadow", "none"),
                p=1.0 
            )
        )
    elif mode == "lcd":
        if gb_palette in ["green", "gray"]:
            hardware_transforms.append(GameBoyFilter(palette=gb_palette, p=1.0))
        elif gb_palette == "full_color":
            hardware_transforms.append(A.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.0, p=1.0))
        else:
            # "random" - randomly choose one of the three LCD aesthetics
            hardware_transforms.append(
                A.OneOf([
                    GameBoyFilter(palette="green", p=1.0),
                    GameBoyFilter(palette="gray", p=1.0),
                    A.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.0, p=1.0)
                ], p=1.0)
            )
    elif mode == "xbrz":
        # Hard limit of 4.0. Anything above 4.0 destroys the geometric data
        # of the 4x upscaled font. Randomizing between 2.0 and 4.0 simulates 
        # varying strengths of the user's emulator smoothing settings.
        import random
        random_scale = random.uniform(2.0, 4.0)
        hardware_transforms.append(
            SmoothUpscale(scale_factor=random_scale, p=1.0)
        )

    return A.Compose([
        # STEP 1: The Screen Hardware (85% chance to apply)
        A.Compose(hardware_transforms, p=0.85),

        # STEP 2: The Signal Degradation (VHS/NTSC style)
        A.OneOf([
            A.RGBShift(r_shift_limit=4, g_shift_limit=4, b_shift_limit=4, p=1.0),
            A.ChannelDropout(channel_drop_range=(1, 1), fill=0, p=1.0),
            A.NoOp() 
        ], p=0.3),

        # STEP 3: The Capture Artifacts
        A.MultiplicativeNoise(multiplier=(0.98, 1.02), per_channel=True, p=0.4),
        A.ImageCompression(quality_range=(70, 95), p=0.7)
    ])
