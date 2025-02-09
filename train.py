import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn, optim
from torchvision import datasets, transforms, models


def arg_parser():
    parser = argparse.ArgumentParser(description="Train a deep learning model.")
    parser.add_argument('--arch', type=str, default="vgg16", help="Model architecture (default: VGG16)")
    parser.add_argument('--save_dir', type=str, default="./checkpoint.pth", help="Directory to save checkpoint")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument('--hidden_units', type=int, default=120, help="Number of hidden units")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--gpu', action="store_true", help="Use GPU if available")
    return parser.parse_args()


def train_transformer():
    return transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def test_transformer():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def data_loader(data_dir, transforms, batch_size=50, shuffle=True):
    dataset = datasets.ImageFolder(data_dir, transform=transforms)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), dataset


def check_gpu(use_gpu):
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
    if device == torch.device("cpu"):
        print("CUDA not found. Using CPU.")
    return device


def load_model(arch="vgg16"):
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_classifier(hidden_units):
    return nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, 90)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(90, 70)),
        ('relu3', nn.ReLU()),
        ('fc4', nn.Linear(70, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))


def validate(model, testloader, criterion, device):
    model.eval()
    loss, accuracy = 0, 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            loss += criterion(outputs, labels).item()
            predictions = torch.exp(outputs).max(dim=1)[1]
            accuracy += (predictions == labels).type(torch.FloatTensor).mean()
    
    return loss / len(testloader), accuracy / len(testloader)


def train_model(model, trainloader, validloader, device, criterion, optimizer, epochs=5):
    print("Starting training...")
    
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        valid_loss, valid_acc = validate(model, validloader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {running_loss/len(trainloader):.4f} | Validation Loss: {valid_loss:.4f} | Validation Accuracy: {valid_acc:.4f}")

    print("Training complete.")
    return model


def test_model(model, testloader, device):
    correct, total = 0, 0
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


def save_checkpoint(model, save_dir, train_data):
    if not isdir(save_dir):
        print(f"Directory {save_dir} not found. Model not saved.")
        return

    checkpoint = {
        'architecture': model.name,
        'classifier': model.classifier,
        'class_to_idx': train_data.class_to_idx,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")
    print(f"Model saved to {save_dir}/checkpoint.pth")


def main():
    args = arg_parser()
    
    data_dir = 'flowers'
    train_dir, valid_dir, test_dir = f"{data_dir}/train", f"{data_dir}/valid", f"{data_dir}/test"
    
    trainloader, train_data = data_loader(train_dir, train_transformer(), shuffle=True)
    validloader, _ = data_loader(valid_dir, test_transformer(), shuffle=False)
    testloader, _ = data_loader(test_dir, test_transformer(), shuffle=False)
    
    device = check_gpu(args.gpu)
    
    model = load_model(args.arch)
    model.classifier = build_classifier(args.hidden_units)
    model.to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    trained_model = train_model(model, trainloader, validloader, device, criterion, optimizer, args.epochs)
    
    test_model(trained_model, testloader, device)
    save_checkpoint(trained_model, args.save_dir, train_data)


if __name__ == '__main__':
    main()
