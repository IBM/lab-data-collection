import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import torch.serialization
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
OUTPUT_FOLDER = "UPDATE_ME/Videoannotation/Data/Video annotation/"


class SimpleVideoClassifier(nn.Module, ):
    ''' A simple video classification model that processes each frame with a CNN
    and averages the predictions.
    '''

    def __init__(self, num_classes, num_frames):
        ''' Initializes the SimpleVideoClassifier:
        num_classes: Number of output classes.
        '''
        super(SimpleVideoClassifier, self).__init__()

        # Load a pre-trained CNN (MobileNetV2)
        # weights = MobileNet_V2_Weights.DEFAULT
        # model = mobilenet_v2(weights=weights)
        self.cnn = torch.hub.load(
            "pytorch/vision:v0.10.0", "mobilenet_v2",
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

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")


def extract_frames_from_video(video_path, num_frames=16, frame_size=224):
    """
    Extract evenly spaced frames from a video.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        frame_size: Size to resize frames to (frame_size x frame_size)

    Returns:
        List of frames as numpy arrays
    """
    if not os.path.exists(video_path):
        print('File missing')
        raise FileNotFoundError
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate indices of frames to extract
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if i in frame_indices:
            # Convert BGR to RGB and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (frame_size, frame_size))
            frames.append(frame)

    cap.release()

    # Pad with last frame if needed
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return frames[:num_frames]


def load_label_encoder(labels):
    """
    Load the label encoder from the training metadata.

    Args:
        labels: List of labels to predict

    Returns:
        LabelEncoder fitted on training labels
    """

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    return label_encoder


def model_loader(model_path, labels, num_frames):
    # Load label encoder
    label_encoder = load_label_encoder(labels)
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")

    # Initialize model
    model = SimpleVideoClassifier(num_classes=num_classes, num_frames=num_frames)
    model.to(device)

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"Model loaded from {model_path}")
    return label_encoder, model


def predict_single_video(video_path, model, label_encoder, num_frames=16):
    """
    Predict the class of a single video.

    Args:
        video_path: Path to the video file
        model_path: Path to the trained model
        label_encoder: List of labels to predict
        num_frames: Number of frames to extract from video

    Returns:
        Dictionary with prediction results
    """

    # Extract frames from video
    #print(f"Extracting frames from {video_path}...")
    frames = extract_frames_from_video(video_path, num_frames=num_frames)

    # Apply transformations (normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Transform frames
    transformed_frames = [transform(frame) for frame in frames]
    video_tensor = torch.stack(transformed_frames)

    # Add batch dimension
    video_tensor = video_tensor.unsqueeze(0).to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        logits = model(video_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class_idx = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0, predicted_class_idx].item()

    # Decode prediction
    predicted_class = label_encoder.classes_[predicted_class_idx]

    # Get all probabilities
    all_probabilities = {}
    for idx, class_name in enumerate(label_encoder.classes_):
        all_probabilities[class_name] = probabilities[0, idx].item()

    result = {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probabilities,
        "class_index": predicted_class_idx
    }
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nAll Class Probabilities:")
    return result
