import json
import random
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import os



def stratified_split_jsonl(
    input_jsonl,
    train_jsonl,
    val_jsonl,
    test_size=0.2,
    random_state=42,
    sample_fraction=1.0,
    stats_file=None
):
    """
    Stratified shuffle split of a JSONL file by label.

    Parameters:
    ----------
    input_jsonl : str
        Path to the input JSONL file
    train_jsonl : str
        Path to the output training JSONL file
    val_jsonl : str
        Path to the output validation JSONL file
    test_size : float, optional
        Proportion of the dataset to include in the validation split (default is 0.2)
    random_state : int, optional
        Random seed for reproducibility (default is 42)
    sample_fraction : float, optional
        Fraction of the dataset to use (default is 1.0, i.e., use all data)
    """

    # --- Load data ---
    records = []
    labels = []

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            records.append(obj)
            labels.append(obj["label"])

    # --- Optionally subsample the dataset ---
    if sample_fraction is None:
        sample_fraction = 1.0

    if not (0 < sample_fraction <= 1.0):
        raise ValueError("sample_fraction must be > 0 and <= 1.0")

    if sample_fraction < 1.0:
        total = len(records)
        k = max(1, int(round(total * sample_fraction)))
        rnd = random.Random(random_state)
        selected = set(rnd.sample(range(total), k))
        records = [records[i] for i in range(total) if i in selected]
        labels = [labels[i] for i in range(total) if i in selected]

    # --- Stratified split ---
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, val_idx = next(splitter.split(records, labels))

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]

    # --- Write output files ---
    for path, data in [(train_jsonl, train_records), (val_jsonl, val_records)]:
        # Use write mode to create/overwrite output JSONL files
        with open(path, "w", encoding="utf-8") as f:
            for obj in data:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Sampled input size: {len(train_records) + len(val_records)}")
    print(f"Train size: {len(train_records)}")
    print(f"Validation size: {len(val_records)}")
    # Optionally write a stats summary to a text file
    if stats_file:
        write_split_stats(stats_file, train_records, val_records, sampled_input_size=len(train_records) + len(val_records))

def augment_jsonl_with_videos(jsonl_path, videos_folder, output_jsonl):
    """
    For each entry in a JSONL file, find video files in a folder that contain
    the 'image' string in their filename and create new entries with the same label
    but the video path as './all_chunks/filename'.

    Parameters
    ----------
    jsonl_path : str
        Path to the input JSONL file.
    videos_folder : str
        Folder containing video files.
    output_jsonl : str
        Path to the output JSONL file (new augmented dataset).
    """
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(jsonl_path)
    if not os.path.isdir(videos_folder):
        raise NotADirectoryError(videos_folder)

    # Gather all video files once
    video_files = os.listdir(videos_folder)

    augmented_lines = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            image_str = entry.get("image")
            label = entry.get("label")

            # Find all videos that contain the image string
            matching_videos = [vf for vf in video_files if image_str in vf]

            for video_name in matching_videos:
                # Prefix with './all_chunks/' as requested
                video_path = "./all_chunks/"+ video_name
                augmented_lines.append({
                    "image": video_path,
                    "label": label
                })

    # Write the augmented JSONL
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for entry in augmented_lines:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def clean_and_update_labels(input_jsonl, output_jsonl=None):
    """
    Read `input_jsonl`, remove entries with label == 'No Action',
    normalize 'Stir'/'Reflux' -> 'Stir or Reflux', and write results
    to `output_jsonl`. If `output_jsonl` is None, replace the original
    file with the cleaned version.
    """
    if not os.path.isfile(input_jsonl):
        raise FileNotFoundError(input_jsonl)

    # Determine destination path
    if output_jsonl:
        dest_path = output_jsonl
        parent = os.path.dirname(dest_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    else:
        dest_path = input_jsonl + ".tmp"

    with open(input_jsonl, "r", encoding="utf-8") as src, \
         open(dest_path, "w", encoding="utf-8") as dst:

        for line in src:
            if not line.strip():
                continue

            obj = json.loads(line)
            label = obj.get("label")

            # Remove unwanted labels
            if label == "No Action":
                continue

            # Normalize labels
            if label in {"Stir", "Reflux"}:
                obj["label"] = "Stir or Reflux"

            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # If we wrote to a temp file, replace the original
    if not output_jsonl:
        os.replace(dest_path, input_jsonl)

def label_distribution(jsonl_path):
    '''
    Get label distribution in a JSONL file.
    Parameters:
    jsonl_path : str
        Path to the JSONL file.
    '''
    counts = Counter()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            counts[json.loads(line)["label"]] += 1
    return counts

def write_split_stats(stats_file, train_records, val_records, test_records, sampled_input_size=None):
    """
    Write a human-readable summary of split statistics to a text file.

    Parameters
    ----------
    stats_file : str
        Path to output text file.
    train_records : list
        List of training record dicts.
    val_records : list
        List of validation record dicts.
    sampled_input_size : int, optional
        Total number of records sampled from the original dataset.
    """
    train_counts = Counter([r.get("label") for r in train_records])
    val_counts = Counter([r.get("label") for r in val_records])
    test_records = Counter([r.get("label") for r in test_records])
    total_counts = train_counts + val_counts

    with open(stats_file, "w", encoding="utf-8") as f:
        if sampled_input_size is not None:
            f.write(f"Sampled input size: {sampled_input_size}\n")
        f.write(f"Train size: {len(train_records)}\n")
        f.write(f"Validation size: {len(val_records)}\n\n")

        f.write("Train label distribution:\n")
        for label, cnt in train_counts.items():
            f.write(f"{label}: {cnt}\n")

        f.write("\nValidation label distribution:\n")
        for label, cnt in val_counts.items():
            f.write(f"{label}: {cnt}\n")

        f.write("\nCombined distribution:\n")
        for label, cnt in total_counts.items():
            f.write(f"{label}: {cnt}\n")


inputfolder = input("Enter path of video dataset folder:")
mainFolder = input("Enter path of main model directory:")
folderName = input("Enter folder name for storage of jsonl splits and model outputs: ")
folder = mainFolder + "\\" + folderName
# Create output folder if it doesn't exist
os.makedirs(folder, exist_ok=True)

# Clean and update labels in the final JSONL file:
'''
- Removes lines with label == 'No Action'
- Replaces 'Stir' and 'Reflux' with 'Stir or Reflux'
'''
# Clean label.jsonl from the input folder and write cleaned version
# into the selected output `folder`.
clean_and_update_labels(os.path.join(inputfolder, "label.jsonl"), os.path.join(folder, "cleanlabel.jsonl"))
stratified_split_jsonl(
    input_jsonl=folder+"\\cleanlabel.jsonl",
    train_jsonl=folder + "\\trainset.jsonl",
    val_jsonl=folder + "\\valtestset.jsonl",
    test_size=0.2,
    random_state=42,
    sample_fraction=1
)
stratified_split_jsonl(
    input_jsonl=folder + "\\valtestset.jsonl",
    train_jsonl=folder + "\\valset.jsonl",
    val_jsonl=folder + "\\testset.jsonl",
    test_size=0.5,
    random_state=42,
    sample_fraction=1.0
)
augment_jsonl_with_videos(folder + "\\trainset.jsonl", inputfolder, folder + "\\train.jsonl")
augment_jsonl_with_videos(folder + "\\valset.jsonl", inputfolder, folder + "\\val.jsonl")
augment_jsonl_with_videos(folder+"\\testset.jsonl", inputfolder, folder + "\\test.jsonl")



print(label_distribution(folder + "\\train.jsonl"))
print(label_distribution(folder + "\\val.jsonl"))
print(label_distribution(folder + "\\test.jsonl"))
