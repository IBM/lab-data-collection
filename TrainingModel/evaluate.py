#!/usr/bin/env python

import datetime
import json
import logging
import math
import uuid
import sys

import cv2
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import torch.serialization
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from train import SimpleVideoClassifier as MyCurrentVideoClassifier

now = datetime.datetime.now()
timestamp_string = now.strftime("%Y%m%d%H%M%S")
rand_str = uuid.uuid4().hex[:8]
input_folder = input('Input the path of the folder containing the trained model:')
video_folder = input('Input the path of the folder containing the dataset used in the training:')
# Create unique log file name
log_file = input_folder+f"eval_info_{timestamp_string}_{rand_str}.log"

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
# Determine device to use (CUDA, MPS, or CPU) and log it
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logging.info("Using MPS (Apple Silicon GPU)!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info("Using CUDA!")
else:
    device = torch.device("cpu")
    logging.info("Using CPU")


class VideoDataset(Dataset):
    def __init__(self, jsonl_file, transform=None, num_frames=16):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = video_folder + self.samples[idx]["image"][1:]
        label = self.encoded_labels[idx]

        # Extract frames from video
        frames = self.extract_frames(video_path)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Stack frames and convert to tensor
        video_tensor = torch.stack(frames)

        return video_tensor, label

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        for i in range(total_frames):
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
            frames.append(frames[-1])

        return frames[: self.num_frames]


class VideoClassifier(nn.Module):
    def __init__(self, num_classes, num_frames=16):
        super(VideoClassifier, self).__init__()

        # Use pretrained CNN for feature extraction
        self.cnn = torch.hub.load(
            "pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True
        )
        # Remove classification head
        self.cnn.classifier = nn.Identity()  

        # Freeze CNN layers (optional) 
        for param in self.cnn.parameters():
            param.requires_grad = False

        # RNN for temporal modeling
        self.rnn = nn.LSTM(
            input_size=1280,  # MobileNetV2 feature size
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # Alternative: Use Transformer instead of RNN
        # self.transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=1280, nhead=8),
        #     num_layers=2
        # )

        # Classification head: fully connected layers, dropout, activation
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 2, 256),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, num_frames, 3, 224, 224)
        batch_size, num_frames, C, H, W = x.shape

        # Extract features for each frame
        cnn_features = []
        for i in range(num_frames):
            frame_features = self.cnn(x[:, i, :, :, :])
            cnn_features.append(frame_features)

        # Stack features: (batch_size, num_frames, feature_size)
        features = torch.stack(cnn_features, dim=1)

        # Temporal modeling with RNN
        rnn_out, (hidden, cell) = self.rnn(features)

        # Use last hidden state from both directions
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # Classification
        output = self.classifier(hidden)

        return output


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for videos, labels in loader:
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100.0 * correct / total


def compute_confusion_matrix(model, loader, device, label_encoder, set_name="Validation", epoch="fig"):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, predicted = outputs.max(1)
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
    
    cm = confusion_matrix(filtered_labels, filtered_preds, labels=valid_indices)
    
    # Use only the available classes for display
    display_labels = [available_classes[i] for i in valid_indices]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap="Blues")
    plt.title(f"{set_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(input_folder+f"{set_name.lower()}_{epoch}_confusion_matrix.png")
    plt.close()
    
    logging.info(f"{set_name} CM - Classes: {display_labels}")
    logging.info(f"CM shape: {cm.shape}")
    
    return cm

def load_model_safely(model_path, device, expected_num_classes=None):
    '''
    Load a PyTorch model checkpoint safely, handling potential serialization issues.
     Attempts to load with weights_only=True first, then with safe globals,
     and finally with weights_only=False as a last resort.
    '''
    try:
        # First try with weights_only=True
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        logging.info("Loaded with weights_only=True")
    except:
        try:
            # If that fails, use safe globals
            import torch.serialization
            torch.serialization.add_safe_globals([sklearn.preprocessing._label.LabelEncoder])
            checkpoint = torch.load(model_path, map_location=device)
            logging.info("Loaded with safe globals")
        except:
            # Last resort: weights_only=False
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            logging.info("Loaded with weights_only=False (use with caution)")

    return checkpoint

if __name__ == "__main__":
    # Data transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Get model checkpoint name from command line argument
    model_check_point_name = sys.argv[1]
    logging.info(f"model_check_point_name={model_check_point_name}")

    

    # Create datasets
    train_dataset = VideoDataset(input_folder + "train.jsonl", transform=transform)
    val_dataset = VideoDataset(input_folder + "val.jsonl", transform=transform)
    test_dataset = VideoDataset(input_folder + "test.jsonl", transform=transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)  # No shuffle for eval
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Load checkpoint
    checkpoint = torch.load(model_check_point_name, map_location=device, weights_only=False)
    
    # Model - create architecture first
    model = MyCurrentVideoClassifier(num_classes=len(train_dataset.label_encoder.classes_))
    model.to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logging.info("Model loaded successfully!")
    if 'val_acc' in checkpoint:
        logging.info(f"Original validation accuracy: {checkpoint['val_acc']:.2f}%")
    if 'epoch' in checkpoint:
        logging.info(f"Trained until epoch: {checkpoint['epoch']}")

    # Comprehensive evaluation function
    def evaluate_model(model, loader, device, label_encoder, set_name="Dataset"):
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for videos, labels in loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm
        }

    # Evaluate on all datasets
    logging.info("\n" + "="*60)
    logging.info("COMPREHENSIVE MODEL EVALUATION")
    logging.info("="*60)

    results = {}
    
    for set_name, loader, dataset in [
        ("TRAIN", train_loader, train_dataset),
        ("VALIDATION", val_loader, val_dataset), 
        ("TEST", test_loader, test_dataset)
    ]:
        logging.info(f"\n{'='*40}")
        logging.info(f"EVALUATING {set_name} SET")
        logging.info(f"{'='*40}")
        
        results[set_name] = evaluate_model(model, loader, device, train_dataset.label_encoder, set_name)
        
        # Print overall metrics
        logging.info(f"Overall Metrics:")
        logging.info(f"  Accuracy:  {results[set_name]['accuracy']:.4f}")
        logging.info(f"  Precision: {results[set_name]['precision']:.4f}")
        logging.info(f"  Recall:    {results[set_name]['recall']:.4f}")
        logging.info(f"  F1-Score:  {results[set_name]['f1']:.4f}")
        
        # Print per-class metrics
        logging.info(f"\nPer-Class Metrics:")
        for i, class_name in enumerate(train_dataset.label_encoder.classes_):
            logging.info(f"  {class_name:20s} | Prec: {results[set_name]['precision_per_class'][i]:.4f} | "
                        f"Rec: {results[set_name]['recall_per_class'][i]:.4f} | "
                        f"F1: {results[set_name]['f1_per_class'][i]:.4f}")

        # Generate confusion matrix plot
        disp = ConfusionMatrixDisplay(confusion_matrix=results[set_name]['confusion_matrix'],
                                      display_labels=train_dataset.label_encoder.classes_)
        disp.plot(cmap=custom_cmap, xticks_rotation='vertical')
        plt.tight_layout()
        plt.savefig(input_folder+f"{set_name.lower()}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed classification report
        report = classification_report(results[set_name]['labels'], results[set_name]['predictions'],
                                       target_names=train_dataset.label_encoder.classes_, digits=4)
        logging.info(f"\nDetailed Classification Report:\n{report}")

    # Summary comparison
    logging.info("\n" + "="*60)
    logging.info("SUMMARY COMPARISON ACROSS ALL DATASETS")
    logging.info("="*60)
    logging.info(f"{'Dataset':<12} | {'Accuracy':<8} | {'Precision':<8} | {'Recall':<8} | {'F1-Score':<8}")
    logging.info(f"{'-'*60}")
    for set_name in [ "TRAIN", "VALIDATION", "TEST"]: 
        metrics = results[set_name]
        logging.info(f"{set_name:<12} | {metrics['accuracy']:.4f}    | {metrics['precision']:.4f}    | "
                    f"{metrics['recall']:.4f}    | {metrics['f1']:.4f}")

    # Detect potential issues
    logging.info("\n" + "="*60)
    logging.info("MODEL PERFORMANCE ANALYSIS")
    logging.info("="*60)
    
    train_acc = results["TRAIN"]['accuracy']
    val_acc = results["VALIDATION"]['accuracy']
    test_acc = results["TEST"]['accuracy']
    
    if train_acc > val_acc + 0.15:  # Significant overfitting
        logging.info("   POTENTIAL OVERFITTING: Train accuracy much higher than validation")
        logging.info(f"   Consider: More regularization, data augmentation, or early stopping")
    
    if val_acc > test_acc + 0.1:  # Poor generalization
        logging.info("   POOR GENERALIZATION: Validation accuracy much higher than test")
        logging.info(f"   Consider: Check data distribution between val/test sets")
    
    # Find worst performing classes
    for set_name in [ "TRAIN","VALIDATION", "TEST"]:
        worst_f1_idx = np.argmin(results[set_name]['f1_per_class'])
        worst_class = train_dataset.label_encoder.classes_[worst_f1_idx]
        worst_f1 = results[set_name]['f1_per_class'][worst_f1_idx]
        
        if worst_f1 < 0.5:
            logging.info(f"  {set_name}: Class '{worst_class}' has low F1-score ({worst_f1:.4f})")
            logging.info(f"  Consider: More samples or data augmentation for this class")

    # Save results to file
    results_summary = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_file': model_check_point_name,
        'results': {
            set_name: {
                'accuracy': float(results[set_name]['accuracy']),
                'precision': float(results[set_name]['precision']),
                'recall': float(results[set_name]['recall']),
                'f1_score': float(results[set_name]['f1']),
                'per_class_f1': {cls: float(score) for cls, score in 
                               zip(train_dataset.label_encoder.classes_, results[set_name]['f1_per_class'])}
            } for set_name in ["TRAIN", "VALIDATION", "TEST"] 
        }
    }
    
    with open(input_folder+"model_evaluation_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    logging.info(f"\n Evaluation complete! Results saved to 'model_evaluation_results.json'")
    logging.info(f" Confusion matrices saved as PNG files")
