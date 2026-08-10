#!/usr/bin/env python

import datetime
import json
import logging
import uuid
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

EPOCHS = 50
FRAMES = 16

now = datetime.datetime.now()
timestamp_string = now.strftime("%Y%m%d%H%M%S")
rand_str = uuid.uuid4().hex[:8]
input_folder = input('Input the path of the folder containing the splits:')
video_folder = input('Input the path of the folder containing the dataset to use:')
# Create a unique log file name
log_file = input_folder+f"info_{timestamp_string}_{rand_str}.log"

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)

# Device configuration (CPU, CUDA, or MPS) and logging
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logging.info("Using MPS (Apple Silicon GPU)!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info("Using CUDA!")
else:
    device = torch.device("cpu")
    logging.info("Using CPU")

# Dataset class
class VideoDataset(Dataset):
    ''' A custom dataset for loading videos and their labels from a JSONL file.
    '''
    def __init__(self, jsonl_file, transform=None, num_frames=FRAMES):
        ''' Initializes the VideoDataset.
    
        jsonl_file: Path to the JSONL file containing video paths and labels.
        transform: Transformations to apply to each frame.
        num_frames: Number of frames to extract from each video.   
        '''
        self.samples = []
        self.transform = transform
        self.num_frames = num_frames

        # Read JSONL file
        with open(jsonl_file, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

        # Encode labels
        self.labels = [sample["label"] for sample in self.samples]
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

    # Return the number of samples
    def __len__(self):
        print("Length of dataset:", len(self.samples))
        return len(self.samples)
    
    # Get a sample by index
    def __getitem__(self, idx):
        video_path = self.samples[idx]["image"]
        label = self.encoded_labels[idx]

        # Extract frames from video
        frames = self.extract_frames(video_folder+ video_path[1:])

        # Apply transformations (e.g. normalization)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Stack frames and convert to tensor
        video_tensor = torch.stack(frames)

        return video_tensor, label

    def extract_frames(self, video_path):
        '''
        Extracts a fixed number of frames from the video at video_path.
        video_path: Path to the video file.
        Returns a list of frames as numpy arrays.
        '''

        print('Extracting frames from:', video_path)
        # Open video file
        cap = cv2.VideoCapture(video_path) 
        frames = []
        # Total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        for i in range(total_frames):
            # print('[' + '-' * i + ' ' * (total_frames - i - 1) + ']')
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                # Convert BGR to RGB and resize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)

        cap.release()

        # If we didn't get enough frames, pad with last frame
        while len(frames) < self.num_frames:
            print(len(frames), self.num_frames)
            frames.append(frames[-1])
        
        # Return only the required number of frames
        return frames[: self.num_frames]


class SimpleVideoClassifier(nn.Module):
    ''' A simple video classification model that processes each frame with a CNN
    and averages the predictions.
    '''
    def __init__(self, num_classes):
        ''' Initializes the SimpleVideoClassifier:
        num_classes: Number of output classes.
        '''
        super(SimpleVideoClassifier, self).__init__()
        
        # Load a pre-trained CNN (MobileNetV2)
        self.cnn = torch.hub.load(
            "pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True
        )
        # Replace the classifier to match the number of classes in the dataset
        self.cnn.classifier[1] = nn.Linear(self.cnn.last_channel, num_classes)

    def forward(self, x):
        ''' Forward pass of the model.
        x: Input tensor of shape (batch_size, num_frames, 3, 224, 224)'''
        # x shape: (batch_size, num_frames, 3, 224, 224)
        batch_size, num_frames = x.shape[0], x.shape[1]

        # Process each frame and average predictions
        frame_logits = []
        for i in range(num_frames):
            frame_out = self.cnn(x[:, i, :, :, :])
            frame_logits.append(frame_out)

        # Average predictions across frames
        avg_logits = torch.mean(torch.stack(frame_logits), dim=0)

        return avg_logits


def train_epoch(model, loader, criterion, optimizer, device):
    ''' Trains the model for one epoch.'''
    # Set model to training mode
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Iterate over data 
    for videos, labels in loader:
        videos, labels = videos.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(videos)
        # Compute loss
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss and accuracy
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


def validate_epoch(model, loader, criterion, device):
    ''' Validates the model for one epoch.'''
    # Set model to evaluation mode
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    # Disable gradient computation
    with torch.no_grad():
        # Iterate over data
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)
            # Forward pass
            outputs = model(videos)
            # Compute loss
            loss = criterion(outputs, labels)
            # Accumulate loss and accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100.0 * correct / total


def compute_confusion_matrix(
    model, loader, device, label_encoder, set_name="Validation", epoch="fig"
):
    ''' Computes and plots the confusion matrix for the given dataset loader.
    '''
    # Set model to evaluation mode
    model.eval()
    # Collect all predictions and true labels
    all_preds = []
    all_labels = []
    # Disable gradient computation
    with torch.no_grad():
        # Iterate over data
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)
            # Forward pass
            outputs = model(videos)
            # Get predicted classes
            _, predicted = outputs.max(1)
            # Accumulate predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Ensure we only include classes that exist in both predictions and true labels
    unique_labels = np.unique(all_labels + all_preds)
    available_classes = label_encoder.classes_

    # Filter to only include classes that exist in our label encoder
    valid_indices = [i for i in unique_labels if i < len(available_classes)]

    # Filter predictions and labels to only include valid classes
    valid_mask = np.isin(all_labels, valid_indices) & np.isin(all_preds, valid_indices)
    filtered_labels = np.array(all_labels)[valid_mask]
    filtered_preds = np.array(all_preds)[valid_mask]

    # Compute confusion matrix
    cm = confusion_matrix(filtered_labels, filtered_preds, labels=valid_indices)

    # Use only the available classes for display
    display_labels = [available_classes[i] for i in valid_indices]

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap="Blues", xticks_rotation="vertical")
    plt.title(f"{set_name} Confusion Matrix")
    plt.tight_layout()
    # Save confusion matrix figure as PNG
    plt.savefig(input_folder+f"{set_name.lower()}_{epoch}_confusion_matrix.png")
    plt.close()

    # Log confusion matrix details
    logging.info(f"{set_name} CM - Classes: {display_labels}")
    logging.info(f"CM shape: {cm.shape}")

    return cm

# Main training loop
if __name__ == "__main__":
    # Data transformations: normalize frames as per ImageNet standards
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    print(input_folder+'train.jsonl')
    train_dataset = VideoDataset(input_folder+"train.jsonl", transform=transform)
    val_dataset = VideoDataset(input_folder+"val.jsonl", transform=transform)

    # Compute class weights automatically from training data to handle class imbalance
    class_counts = np.bincount(train_dataset.encoded_labels)
    total_samples = len(train_dataset.encoded_labels)
    weights = torch.FloatTensor([total_samples / count for count in class_counts]).to(
        device
    )

    # Log class distribution and weights
    logging.info(f"Class distribution: {class_counts}")
    logging.info(f"Class weights: {weights}")
    logging.info(f"Classes: {train_dataset.label_encoder.classes_}")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Model initialization
    model = SimpleVideoClassifier(num_classes=len(train_dataset.label_encoder.classes_))
    # Move model to the appropriate device
    model.to(device)

    # Loss and optimizer with weighted loss
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = EPOCHS
    
    # Training loop
    best_acc = 0
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        # Validate for one epoch
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Log epoch results
        logging.info(f"Epoch {epoch + 1}/{num_epochs}:")
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            # torch.save(model.state_dict(), f"best_model_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "label_encoder": train_dataset.label_encoder,
                    "num_classes": len(train_dataset.label_encoder.classes_),
                },
                input_folder+f"best_model_epoch_{epoch}.pth",
            )

        # Compute confusion matrices for train and validation sets and log them
        cm = compute_confusion_matrix(
            model,
            train_loader,
            device,
            train_dataset.label_encoder,
            set_name="Train",
            epoch=str(epoch),
        )
        logging.info(f" cm Train {cm}")

        cm = compute_confusion_matrix(
            model,
            val_loader,
            device,
            train_dataset.label_encoder,
            set_name="Validation",
            epoch=str(epoch),
        )
        logging.info(f" cm Validation {cm}")
        # Update learning rate
        scheduler.step()

    # After training, compute confusion matrices for train, validation
    logging.info("Generating confusion matrices...")

    cm = compute_confusion_matrix(
        model,
        train_loader,
        device,
        train_dataset.label_encoder,
        set_name="Train",
        epoch="finished",
    )
    logging.info(f" cm Train {cm}")

    cm = compute_confusion_matrix(
        model,
        val_loader,
        device,
        train_dataset.label_encoder,
        set_name="Validation",
        epoch="finished",
    )
    logging.info(f" cm Validation {cm}")

    # Create test dataset and loader
    test_dataset = VideoDataset(input_folder+"test.jsonl", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Compute confusion matrix for test set
    compute_confusion_matrix(
        model, test_loader, device, test_dataset.label_encoder, set_name="Test"
    )
 