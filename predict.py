import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
from train import check_gpu
from torchvision import models


def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image', type=str, help='Point to image file for prediction.', required=True)
    parser.add_argument('--checkpoint', type=str, help='Point to checkpoint file.', required=True)
    parser.add_argument('--top_k', type=int, help='Choose top K matches.', default=5)
    parser.add_argument('--category_names', type=str, help='Path to category names file.', default='cat_to_name.json')
    parser.add_argument('--gpu', type=str, help='Use GPU for inference if available.', default="gpu")

    return parser.parse_args()


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    
    for param in model.parameters():
        param.requires_grad = False

    # Load from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    return model


def process_image(image_path):
    img = PIL.Image.open(image_path)

    original_width, original_height = img.size

    if original_width < original_height:
        size = (256, int(256 * original_height / original_width))
    else:
        size = (int(256 * original_width / original_height), 256)

    img.thumbnail(size)

    center_x, center_y = img.size[0] / 2, img.size[1] / 2
    left, top = center_x - 112, center_y - 112
    right, bottom = center_x + 112, center_y + 112
    img = img.crop((left, top, right, bottom))

    numpy_img = np.array(img) / 255

    # Normalize each color channel
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    numpy_img = (numpy_img - mean) / std

    # Transpose to match PyTorch format (C, H, W)
    numpy_img = numpy_img.transpose(2, 0, 1)

    return numpy_img


def predict(image_path, model, device, cat_to_name, top_k=5):
    model.to(device)
    model.eval()

    image_tensor = torch.from_numpy(process_image(image_path)).type(torch.FloatTensor).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model.forward(image_tensor)

    probabilities = torch.exp(output)
    top_probs, top_labels = probabilities.topk(top_k)

    top_probs = top_probs.cpu().numpy().flatten()
    top_labels = top_labels.cpu().numpy().flatten()

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[label] for label in top_labels]
    top_flowers = [cat_to_name[label] for label in top_labels]

    return top_probs, top_labels, top_flowers


def print_probability(probs, flowers):
    # Converts two lists into a dictionary to print on screen
    for i, (flower, prob) in enumerate(zip(flowers, probs)):
        print(f"Rank {i+1}: Flower: {flower}, Likelihood: {ceil(prob * 100)}%")


def main():
    args = arg_parser()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)

    device = check_gpu(gpu_arg=args.gpu)

    top_probs, top_labels, top_flowers = predict(args.image, model, device, cat_to_name, args.top_k)

    print_probability(top_probs, top_flowers)


if __name__ == '__main__':
    main()
