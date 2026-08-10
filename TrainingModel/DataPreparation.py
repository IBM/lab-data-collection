import cv2
import os
from collections import defaultdict
import re
import json

SECONDS = 4
FILENAME_RE = re.compile(
    r"^Cam(?P<cam>\d+)_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<time>\d{2}-\d{2}-\d{2})_"
    r"chunk_(?P<chunk>\d+)"
    r"\.(mp4|avi)$"
)


VIDEO_FOLDER = input('Add path of folder containing not chunked video')
CHUNKS_FOLDER = input('Add path of folder containing chunked video')
OUTPUT_FOLDER = input('Add path of destination folder')

def index_videos(folder):
    '''
    Index videos in a folder based on their metadata extracted from filenames.
    Parameters:
        folder : str
    '''
    index = defaultdict(dict)

    for name in os.listdir(folder):
        match = FILENAME_RE.match(name)
        if not match:
            continue

        key = (
            match["date"],
            match["time"],
            match["chunk"]
        )

        cam_id = match["cam"]
        index[key][cam_id] = os.path.join(folder, name)

    return index


def make_output_name(cam1_path):
    '''
    Generate output filename for combined CamBoth video.
    Parameters:
        cam1_path : str
    '''
    base = os.path.basename(cam1_path)
    return base.replace("Cam1_", "CamBoth_")


def find_cam_pairs(folder):
    '''
    Find pairs of Cam1 and Cam2 videos in a folder.
    Parameters:
        folder : str
    '''
    index = index_videos(folder)

    pairs = []
    for key, cams in index.items():
        if "1" in cams and "2" in cams:
            pairs.append(
                (cams["1"], cams["2"])
            )
    return pairs


def append_jsonl_file(jsonl_path, output_jsonl):
    """
    Append all JSONL files from a folder into an existing JSONL file.
    Parameters
    ----------
    jsonl_path : str
        Folder containing .jsonl files
    output_jsonl : str
        Path to the target JSONL file (already created or new)
    """
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(jsonl_path)

    if os.path.isdir(output_jsonl):
        raise ValueError(f"Output path is a directory: {output_jsonl}")

    with open(output_jsonl, "a", encoding="utf-8") as out_f:
        with open(jsonl_path, "r", encoding="utf-8") as in_f:
            for line in in_f:
                if line.strip():
                    out_f.write(line)


def remove_jsonl_by_field(file_path, field, value):
    '''
    Remove entries from a JSONL file where a specific field matches a given value.
    Parameters: 
        file_path : str
            Path to the JSONL file to process.
        field : str
            The field to check in each JSON object.
        value : any
            The value to match for removal.
    '''
    temp_path = file_path + ".tmp"

    with open(file_path, "r", encoding="utf-8") as src, \
         open(temp_path, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue

            obj = json.loads(line)
            if obj.get(field) != value:
                dst.write(line)

    os.replace(temp_path, file_path)


def split_video_opencv(
    input_video: str,
    output_dir: str,
    chunk_duration: int = 4
):
    ''' Split a video into chunks of specified duration using OpenCV.
    Parameters:
        input_video : str
            Path to the input video file.
        output_dir : str
            Directory to save the output video chunks.
        chunk_duration : int
            Duration of each chunk in seconds.
    '''
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate number of frames per chunk
    frames_per_chunk = int(fps * chunk_duration)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    chunk_index = 0
    frame_count = 0
    writer = None

    # Read and write frames to chunks
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Start a new chunk if needed
        if frame_count % frames_per_chunk == 0:
            if writer is not None:
                writer.release()

            # Create new output file path
            output_path = os.path.join(
                output_dir, f"{os.path.basename(input_video[:-4])}_chunk_{chunk_index}.mp4"
            )

            # Initialize VideoWriter for the new chunk
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            chunk_index += 1

        writer.write(frame)
        frame_count += 1

    if writer is not None:
        writer.release()

    cap.release()


def concatenate_videos_opencv(video1, video2, output_path):
    '''
    Concatenate two videos one after the other using OpenCV.
    Parameters:
    video1 : str
        Path to the first video file.
    video2 : str
        Path to the second video file.
    output_path : str
        Path to save the concatenated video.
    '''
    # Validate inputs
    if not os.path.isfile(video1):
        raise FileNotFoundError(video1)
    if not os.path.isfile(video2):
        raise FileNotFoundError(video2)

    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    if not cap1.isOpened():
        raise RuntimeError(f"Failed to open video: {video1}")
    if not cap2.isOpened():
        raise RuntimeError(f"Failed to open video: {video2}")

    # Read properties from first video
    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Validate second video matches
    if int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)) != width or \
       int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)) != height:
        raise ValueError("Videos must have the same resolution")

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write first video
    while True:
        ret, frame = cap1.read()
        if not ret:
            break
        writer.write(frame)

    # Write second video
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        writer.write(frame)

    cap1.release()
    cap2.release()
    writer.release()


def stack_videos_vertical_opencv(video1, video2, output_path):
    '''
    Stack two videos vertically (one on top of the other) using OpenCV.

    Parameters:
    video1 : str
        Path to the first video file (top).
    video2 : str
        Path to the second video file (bottom).
    output_path : str
        Path to save the stacked video.
    '''

    # Validate inputs
    if not os.path.isfile(video1):
        raise FileNotFoundError(video1)
    if not os.path.isfile(video2):
        raise FileNotFoundError(video2)

    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    if not cap1.isOpened():
        raise RuntimeError(f"Failed to open video: {video1}")
    if not cap2.isOpened():
        raise RuntimeError(f"Failed to open video: {video2}")

    # Read properties
    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Validate resolution match
    if int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)) != width or \
       int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)) != height:
        raise ValueError("Videos must have the same resolution")
    out_size = (width, height * 2)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_path, fourcc, fps, out_size)

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Stop if either video ends
        if not ret1 or not ret2:
            break

        # Stack vertically
        stacked_frame = cv2.vconcat([frame1, frame2])
        # alternatively: np.vstack((frame1, frame2))

        writer.write(stacked_frame)

    cap1.release()
    cap2.release()
    writer.release()


def deduplicate_jsonl_exact(file_path):
    '''
    Remove exact duplicate lines from a JSONL file.
    Parameters:
        file_path : str
            Path to the JSONL file to deduplicate.
    '''
    temp_path = file_path + ".tmp"
    seen = set()
    # Read original file and write unique lines to temp file
    with open(file_path, "r", encoding="utf-8") as src, \
         open(temp_path, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.rstrip("\n")
            if not line or line in seen:
                continue
            seen.add(line)
            dst.write(line + "\n")
    os.replace(temp_path, file_path)


def chuncking_videos_and_appending_jsonl(video_folder, chunks_folder, seconds):
    # Split videos and append JSONL files
    for video_folder in os.listdir(os.path.join(video_folder)):
        for vid in os.listdir(os.path.join(video_folder,video_folder)):
            if vid.lower().endswith('.mp4'):
                # Split video into chunks of the specified SECONDS
                try:
                    split_video_opencv(os.path.join(video_folder,video_folder, vid), chunks_folder, seconds)
                except (FileNotFoundError, RuntimeError) as e:
                    print("Error:", e)
                    pass
            elif vid.lower().endswith('.jsonl'):
                # Append JSONL file to the output JSONL
                append_jsonl_file(os.path.join(video_folder,video_folder, vid), os.path.join(chunks_folder,"label.jsonl"))
            print(vid)
    # Deduplicate the final JSONL file
    deduplicate_jsonl_exact(os.path.join(chunks_folder,"label.jsonl"))


def concatenate_videos_and_jsonl(chunks_folder, output_folder):
    # Merge Cam1 and Cam2 videos into CamBoth videos and save them into all_chunks folder
    for cam1, cam2 in find_cam_pairs(chunks_folder):
        # Generate output filename
        output = make_output_name(cam1)
        concatenate_videos_opencv(cam1,
                                cam2,
                                os.path.join(output_folder, output))
        print(f'{cam1, cam2} merged!')


def stack_videos_and_jsonl(chunks_folder, output_folder):
    for cam1, cam2 in find_cam_pairs(chunks_folder):
        # Generate output filename
        output = make_output_name(cam1)
        print(output)
        stack_videos_vertical_opencv(cam1,
                                cam2,
                                os.path.join(output_folder, output))
    # Move JSONL entries from main folder to all_chunks folder
    append_jsonl_file(os.path.join(chunks_folder, "label.jsonl"), os.path.join(output_folder, "label.jsonl"))


if VIDEO_FOLDER != "":
    chuncking_videos_and_appending_jsonl(VIDEO_FOLDER, CHUNKS_FOLDER, SECONDS)
# Run the preferred version of video combination (concatenation or stacking)
concatenate_videos_and_jsonl(CHUNKS_FOLDER, OUTPUT_FOLDER)
#stack_videos_and_jsonl(CHUNKS_FOLDER, OUTPUT_FOLDER)
