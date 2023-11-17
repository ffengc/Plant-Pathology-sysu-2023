from PIL import Image, ImageEnhance, ImageOps
import random

def apply_randaugment(image):
    # RandAugment hyperparameters
    n_ops = 2  # Number of augmentation operations to apply
    magnitude = 10  # Magnitude of the applied operations

    for _ in range(n_ops):
        op = random.choice([
            'autocontrast', 'equalize', 'rotate', 'posterize',
            'solarize', 'color', 'contrast', 'brightness',
            'sharpness', 'shearX', 'shearY', 'translateX', 'translateY'
        ])

        if op == 'autocontrast':
            image = ImageOps.autocontrast(image)
        elif op == 'equalize':
            image = ImageOps.equalize(image)
        elif op == 'rotate':
            degrees = random.uniform(-magnitude, magnitude)
            image = image.rotate(degrees)
        elif op == 'posterize':
            bits = random.randint(1, 8)
            image = ImageOps.posterize(image, bits)
        elif op == 'solarize':
            threshold = random.uniform(0, 256)
            image = ImageOps.solarize(image, threshold)
        elif op == 'color':
            factor = random.uniform(1 - magnitude / 100, 1 + magnitude / 100)
            image = ImageEnhance.Color(image).enhance(factor)
        elif op == 'contrast':
            factor = random.uniform(1 - magnitude / 100, 1 + magnitude / 100)
            image = ImageEnhance.Contrast(image).enhance(factor)
        elif op == 'brightness':
            factor = random.uniform(1 - magnitude / 100, 1 + magnitude / 100)
            image = ImageEnhance.Brightness(image).enhance(factor)
        elif op == 'sharpness':
            factor = random.uniform(1 - magnitude / 100, 1 + magnitude / 100)
            image = ImageEnhance.Sharpness(image).enhance(factor)
        elif op == 'shearX':
            shear_factor = random.uniform(-magnitude, magnitude)
            image = image.transform(image.size, Image.AFFINE, (1, shear_factor, 0, 0, 1, 0))
        elif op == 'shearY':
            shear_factor = random.uniform(-magnitude, magnitude)
            image = image.transform(image.size, Image.AFFINE, (1, 0, 0, shear_factor, 1, 0))
        elif op == 'translateX':
            translate_factor = random.uniform(-magnitude, magnitude)
            image = image.transform(image.size, Image.AFFINE, (1, 0, translate_factor, 0, 1, 0))
        elif op == 'translateY':
            translate_factor = random.uniform(-magnitude, magnitude)
            image = image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, translate_factor))
    return image

# Example usage:
input_image_path = "/home/hbenke/Project/Yufc/Project/cv/Data/plant-pathology-data/train/images/8a1a97abda0b4a7a.jpg"
output_image_path = "/home/hbenke/Project/Yufc/Project/cv/plant-Pathology-main/utils/output.png"
output_image_path_org = "/home/hbenke/Project/Yufc/Project/cv/plant-Pathology-main/utils/org.png"

# Load the input image
input_image = Image.open(input_image_path)

# Apply RandAugment
augmented_image = apply_randaugment(input_image)
# Save the augmented image
augmented_image.save(output_image_path)
input_image.save(output_image_path_org)


# /home/hbenke/Project/Yufc/Project/cv/plant-Pathology-main/utils/output.png
# /home/hbenke/Project/Yufc/Project/cv/Data/plant-pathology-data/train/images/8a1a97abda0b4a7a.jpg