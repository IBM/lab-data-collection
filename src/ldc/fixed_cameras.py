import os
import sys
import threading
import time
import customtkinter as tk
import tkinter as tki
from functools import partial
import json
from datetime import datetime
import re, ast

import cv2
import numpy as np
from PIL import Image
from yoctopuce.yocto_api import YAPI, YRefParam
from yoctopuce.yocto_lightsensor import YLightSensor
from ldc.VideoClassification import model_loader, predict_single_video
import requests

from ldc.frame_manipulation import process_fixedframe
from ldc.utils import LightMotion_sensor, note_extractor, start_record
from ldc.config_manager import save_config


# Variables
FRAME_RATE = 10
FRAME_INTERVAL = 1 / FRAME_RATE
TEMP_ACQ_INTERVAL = 5

NUM_FRAMES = 16
FRAME_SIZE = 224
CLASSES = ['Addition of liquids', 'Extraction', 'Filtration', 'Inert atmosphere', 'Stir or Reflux']
model = None
label_encoder = None
video_predictions = {}


class GraphicUI:
    def __init__(self, config, root_title="VideoManager", top_camera=2, bottom_camera=3):   # Define class
        self.root = tk.CTk()
        # Create GUI main window
        self.root.title(root_title)    # Title of the main window
        self.config = config
        # cv2.VideoCapture(top_camera)    # Open camera 1 (Top position)
        self.cap1 = top_camera
        # cv2.VideoCapture(bottom_camera)  # Open camera 2 (Bottom position)
        self.cap2 = bottom_camera
        self.variables = {'one': None, 'two': None, 'rec': False, 'fl': None, 'gr': None, 'rec_off': None}
        self.frame = None               # variable containing Frame camera 1
        self.frame2 = None              # variable containing Frame camera 1
        self.selected_camera = 1        # variable to handle the shown frame
        self.reactions = {}             # dictionary containing reactions for annotations: {'id': 'Status'}
        self.rec_event = threading.Event()
        self.running = {'run': True}

        # Open Yocto API to connect light sensor
        errmsg = YRefParam()
        if YAPI.RegisterHub(self.config['URL SENSORS'], errmsg) != YAPI.SUCCESS:
            sys.exit("init error" + errmsg.value)
        self.sensor = YLightSensor.FirstLightSensor()       # open light sensor
        self.menubar = tki.Menu(self.root)
        self.settings_menu = tki.Menu(self.menubar, tearoff=0)
        self.settings_menu.add_command(label="General", command=self.open_settings)
        self.settings_menu.add_separator()
        self.menubar.add_cascade(label="Settings", menu=self.settings_menu)
        self.root.config(menu=self.menubar)

        # Define variable to show if recording is active or not
        self.var = tk.StringVar()
        self.var.set("NOT recording")

        # GUI structure and widgets
        label = tk.CTkLabel(self.root, textvariable=self.var)  # Show Recording/Not Recording
        label.grid(column=0, row=0)

        # Buttons to change view between top and bottom camera
        button3 = tk.CTkButton(self.root, text="Top Camera", command=self.top_cam)
        button3.grid(column=1, row=0)

        button4 = tk.CTkButton(self.root, text="Bottom Camera", command=self.bottom_cam)
        button4.grid(column=2, row=0)

        # Panel showing frames
        self.panel = tk.CTkLabel(self.root)
        self.panel.grid(column=0, row=1, columnspan=3)

        # Threading event that will start the thread when specified
        self.thread_start = threading.Event()

        # Video classification thread
        thread = threading.Thread(target=self.video_classification, daemon=True)
        # Annotation extraction thread (While loop every 4 minutes)
        thread1 = threading.Thread(target=self.annotation_extractor, daemon=True)
        # Record frames thread (10 hz loop)
        record = threading.Thread(target=self.record_frames, daemon=True)

        # Start the threads
        record.start()
        thread.start()
        thread1.start()

        # wait that the record thread loads the frames changing the value of frame and frame2 from None
        time.sleep(5)

        # Recursive function that shows the frame in the panel of the GUI
        self.show_frame()

        # release of the cameras and of all the windows
        self.cap1.release()
        self.cap2.release()

        record.join()
        thread.join()
        cv2.destroyAllWindows()

    # Function to select top camera to show
    def top_cam(self):
        self.selected_camera = 1

    # Function to select bottom camera to show
    def bottom_cam(self):
        self.selected_camera = 2


    def open_settings(self):

        popup = tk.CTkToplevel(self.root)
        popup.title("Settings")
        popup.geometry("400x300")
        popup.grab_set()

        entries = {}

        save_btn = tk.CTkButton(
            popup,
            text="Save",
            state="disabled"
        )

        def check_changes(*args):
            modified = False
            for key, entry in entries.items():
                if entry.get() != str(self.config[key]):
                    modified = True
                    break
            save_btn.configure(
                state="normal" if modified else "disabled"
            )
        row = 0
        for key, value in self.config.items():
            tk.CTkLabel(
                popup,
                text=key
            ).grid(row=row, column=0, padx=10, pady=5, sticky="w")

            entry = tk.CTkEntry(popup, width=200)
            entry.insert(0, str(value))
            entry.grid(row=row, column=1, padx=10, pady=5)
            entry.bind("<KeyRelease>", check_changes)
            entries[key] = entry
            row += 1

        def save():

            new_config = {}

            for key, entry in entries.items():

                value = entry.get()

                old_value = self.config[key]

                if isinstance(old_value, bool):
                    value = value.lower() == "true"

                elif isinstance(old_value, int):
                    value = int(value)

                elif isinstance(old_value, float):
                    value = float(value)

                new_config[key] = value

            new_config['COOKIES'] = d = ast.literal_eval(new_config['COOKIES'])
            save_config(new_config)
            self.config = new_config

            save_btn.configure(state="disabled")

            popup.destroy()

        save_btn.configure(command=save)
        save_btn.grid(row=row, column=0, columnspan=2, pady=20)
        
    # Show frame loop (Main loop)
    def show_frame(self):
        while self.running['run']:
            if self.selected_camera == 1:
                # Camera 1
                frame_show = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)  # convert frame in BGR into RGB
                img = Image.fromarray(frame_show)               # create the image from the array and show in the panel
                imgtk1 = tk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
                self.panel.imgtk = imgtk1
                self.panel.configure(image=imgtk1, text="")
            elif self.selected_camera == 2:
                # Camera 2
                frame_show2 = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2RGB)   # convert frame in BGR into RGB
                img2 = Image.fromarray(frame_show2)             # create the image from the array and show in the panel
                imgtk2 = tk.CTkImage(light_image=img2, dark_image=img2, size=(640, 480))
                self.panel.imgtk = imgtk2
                self.panel.configure(image=imgtk2, text="")
            # Controls the light sensor output and implement a motion sensor on camera 1
            LightMotion_sensor(frame=self.frame,
                               sensor=self.sensor,
                               variables=self.variables,
                               rec='rec',
                               thread_start=self.thread_start,
                               rec_off='rec_off',
                               one='one',
                               two='two',
                               root=self.root,
                               fl='fl',
                               gr='gr')
            # saves the current gray frame as the last one (for motion sensor)
            self.variables['fl'] = self.variables['gr']
            if self.variables['rec'] is True:   # handle variable that show if the system is recording or not
                self.var.set("Recording")
            else:
                self.var.set("NOT recording")
            self.root.update()          # update the panel and the variable var

# Record frames Loop (Thread: record)
    def record_frames(self):
        # global model, label_encoder, predictions
        frame_count = 0
        while self.running['run']:
            start_time = time.time()            # checks the starting time of the cycle
            ret, frame = self.cap1.read()       # read from camera 1
            ret2, frame2 = self.cap2.read()     # read from camera 2
            if ret and ret2:
                frame = cv2.rotate(frame, cv2.ROTATE_180)       # rotates the video from camera 1
                frame2 = cv2.rotate(frame2, cv2.ROTATE_180)  # rotates the video from camera 2
                frame_show = process_fixedframe(frame, 1)   # write time and camera number on the frame
                frame_show2 = process_fixedframe(frame2, 2)  # write time and camera number on the frame
                # saves frames as variables to transfer them to the show_frame()
                self.frame = frame_show
                self.frame2 = frame_show2
                if self.variables['rec']:   # if is_recording is True saves the frames in the right file
                    self.rec_event.set()
                    if frame_count % 39 == 0:
                        self.variables['one'].release()
                        self.variables['two'].release()
                        if self.variables['rec']:
                            start_record(variables=self.variables,
                                         rec='rec',
                                         root=False,
                                         one='one',
                                         two='two')
                            frame_count = 0
                    self.variables['one'].write(frame_show)
                    self.variables['two'].write(frame_show2)
                    frame_count += 1
                elapsed_time = time.time() - start_time     # calculates how long did it take to run the cycle
                # calculates the time to wait to have the right frequency in the frame acquisition
                time_to_wait = max(0, FRAME_INTERVAL - elapsed_time)
                time.sleep(time_to_wait)                    # waits until the full cycle takes 0.1 s before restarting
            else:
                time.sleep(10)

    # annotation extractor Loop (Thread: thread1)
    def annotation_extractor(self):
        r = requests.get(self.config['URL'], cookies=self.config["COOKIES"])
        data = r.json()
        for i in data['reactions']:
            if i['status'] != "Successful" and i['status'] != "Not Successful":
                self.reactions[i['id']] = i['status']
        while self.running['run']:  # Enters loop
            start_time = time.time()  # checks the starting time of the cycle
            try:
                r = requests.get(self.config['URL'], cookies=self.config["COOKIES"])
                data = r.json()
                for i in data['reactions']:
                    if (i['id'] not in self.reactions.keys() and i['status'] != "Successful"
                            and i['status'] != "Not Successful"):
                        self.reactions[i['id']] = i['status']
                    elif i['id'] in self.reactions.keys() and (self.reactions[i['id']] != i['status'] and (i['status'] == "Successful" or i['status'] == "Not Successful")):
                        self.reactions[i['id']] = i['status']
                        # create dictionary for annotation of the completed reaction
                        # in the shape {'date':['[date-time]: note1', '[date-time]: note2']}
                        annotations = {}
                        # take description of the reaction and
        # converts the quill dictionary into a dictionary with form {'description': 'description... (str)'}
                        r1 = requests.get(re.sub(r"(reactions).*", r"\1" + '/', self.config['URL'])+f'{i['id']}.json', cookies=self.config["COOKIES"])
                        data1 = r1.json()
                        note = note_extractor(data1['reaction'])
                        # takes the string 'description... and splits where a '[' is found and create a list of strings.
                        # Loops on the list skipping the first element
                        for j in note["description"].split('Procedure:')[-1].split("[")[1:]:
                            j = "[" + j                                     # puts back the removed '[' in each element.
                            # if the date of the split element is not in the keys of annotations dict:
                            if j[1:11] not in annotations.keys():
                                # adds the element to the dictionary: 'date':['[date-time]: note1']
                                annotations[j[1:11]] = [j]
                            # if the date is already a key of the annotations dict:
                            elif j[1:11] in annotations.keys():
                                # append the note to the list ['[date-time]: note1', '[date-time]: note2']
                                annotations[j[1:11]].append(i)
                        for date, note in annotations.items():           # Open dictionary and loop for date and [notes]
                            # creates path of the folder named for the relative date
                            fdate = f"{date[6:]}-{date[3:5]}-{date[0:2]}"
                            obj_path = os.path.join(self.config['OUTPUT_FOLDER'], fdate)
                            # creates a list of files contained in the folder /date
                            # os.makedirs(obj_path, exist_ok=True)
                            files = os.listdir(obj_path)
                            if "notes.txt" in files:                      # if the folder already contains a notes.txt:
                                # open notes.txt
                                with (open(os.path.join(self.config['OUTPUT_FOLDER'], fdate, "notes.txt"), "r", encoding="utf-8")
                                      as file):
                                    content = file.read()                              # read text in the file
                                    # splits the text by '[' and loops over the list of strings obtained
                                    for k in content.split("["):
                                        k = "[" + k                                       # add back '['
                                        note.append(k)                                    # append string to [notes]
                                    # sort the strings in [notes] by time and date
                                    sorted_logs = sorted(note, key=lambda entry: entry[12:20])
                                    # save the sorted [notes] in the file 'notes.txt'
                                with (open(os.path.join(self.config['OUTPUT_FOLDER'], fdate, "notes.txt"), "w", encoding="utf-8")
                                      as file):
                                    file.write("\n".join(sorted_logs))
                                    print('note added to the file text')

                            else:                                           # if the folder does not contain a file note
                                # creates a file 'notes.txt'
                                with (open(os.path.join(self.config['OUTPUT_FOLDER'], fdate, "notes.txt"), "w", encoding="utf-8")
                                      as file):
                                    # sort the strings in [notes] by time and date
                                    sorted_logs = sorted(note, key=lambda entry: entry[12:20])
                                    # save the sorted [notes] in the file 'notes.txt'
                                    file.write("\n".join(sorted_logs))
                                    print('File text created')

            except requests.exceptions.ReadTimeout:
                print("Request timed out. Retrying or increasing timeout might help.")
                self.annotation_extractor()
            except requests.exceptions.ConnectionError:
                print("Failed to connect. Check the server or your internet connection.")
                self.annotation_extractor()
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
                self.annotation_extractor()

                # since a completed reaction was found,
                # the function is called back to exit the loop and save the new status

            elapsed_time = time.time() - start_time         # calculates how long did it take to run the cycle
            print(elapsed_time)
            time_to_wait = max(0, TEMP_ACQ_INTERVAL - elapsed_time)           # calculates the time to wait
            time.sleep(time_to_wait)                            # waits 2 minutes before restarting

    def video_classification(self):
        global label_encoder, model
        label_encoder, model = model_loader(model_path=self.config['MODEL_PATH'], labels=CLASSES, num_frames=NUM_FRAMES)
        while self.running['run']:
            self.rec_event.wait()
            current_datetime = datetime.now().strftime("%Y-%m-%d")
            files = os.listdir(os.path.join(self.config['OUTPUT_FOLDER'], f"{current_datetime[0:10]}"))
            if len(files) > 2 and len(files) % 2 == 0:
                self.combine_and_predict_most_recent(current_datetime, files, 4, model, label_encoder, self.rec_event)

    def average_scores_by_label(self, data: dict) -> dict:
        sums = {}
        counts = {}

        for _, file_scores in data.items():
            for label, value in file_scores['all_probabilities'].items():
                sums[label] = sums.get(label, 0.0) + value
                counts[label] = counts.get(label, 0) + 1

        return {list(data.keys())[0]: {label: sums[label] / counts[label] for label in sums}}

    def concatenate_videos_opencv(self, video1, video2, output_path):
        # Validate inputs
        if not os.path.isfile(video1):
            raise FileNotFoundError(video1)
        if not os.path.isfile(video2):
            raise FileNotFoundError(video2)

        cap1 = cv2.VideoCapture(video1)
        cap2 = cv2.VideoCapture(video2)

        if not cap1.isOpened():
            return
        if not cap2.isOpened():
            return

        # Read properties from first video
        fps = cap1.get(cv2.CAP_PROP_FPS)
        width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Validate second video matches
        if int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)) != width or \
                int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)) != height:
            raise ValueError("Videos must have the same resolution")

        if abs(cap2.get(cv2.CAP_PROP_FPS) - fps) > 0.01:
            print(os.path.basename(output_path)[8:27] + "do not have the same FPS")
            return

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

    def combine_and_predict_most_recent(self, current_datetime, files, n, model, label_encoder, event):
        global video_predictions
        files_with_time = [(f, os.path.getmtime(os.path.join(self.config['OUTPUT_FOLDER'], f"{current_datetime[0:10]}", f)))
                           for f in files]
        files_with_time.sort(key=lambda x: x[1], reverse=False)
        recent_files = [os.path.join(self.config['OUTPUT_FOLDER'], f"{current_datetime[0:10]}", f[0]) for f in
                        files_with_time[-n:]]
        for vid in recent_files:
            base = os.path.basename(vid)
            if base[:5] == "Cam1_":
                output = base.replace("Cam1_", "CamBoth_")
                if not os.path.exists(os.path.join(self.config['OUTPUT_FOLDER'], "DirectPrediction", output)):
                    vid1 = vid.replace("Cam1_", "Cam2_")
                    cap = cv2.VideoCapture(vid)
                    cap1 = cv2.VideoCapture(vid1)
                    if cap.isOpened() and cap1.isOpened():
                        self.concatenate_videos_opencv(vid, vid1,
                                                  os.path.join(self.config['OUTPUT_FOLDER'], "DirectPrediction", output))
                        video_predictions[output] = "Placeholder"
                    cap.release()
                    cap1.release()
        output = os.path.basename(recent_files[2].replace("Cam1_", "CamBoth_"))
        print(output)
        if len(video_predictions) != 0:
            for video in video_predictions.keys():
                cap = cv2.VideoCapture(os.path.join(self.config['OUTPUT_FOLDER'], "DirectPrediction",video))
                if video_predictions[video] == "Placeholder" and cap.isOpened():
                    cap.release()
                    print(3)
                    result = predict_single_video(
                        video_path=os.path.join(self.config['OUTPUT_FOLDER'], "DirectPrediction", video),
                        model=model,
                        label_encoder=label_encoder,
                        num_frames=NUM_FRAMES
                        )
                    for class_name, prob in result['all_probabilities'].items():
                        print(f"  {class_name}: {prob:.4f}")
                    video_predictions[video] = result
                    print(f"Predicted Class: {result['predicted_class']}")
        if os.path.exists(os.path.join(self.config['OUTPUT_FOLDER'], "DirectPrediction", output)):
            average = self.average_scores_by_label(video_predictions)
            values = np.array(list(average[list(video_predictions.keys())[0]].values()))
            # Argmax
            max_idx = np.argmax(values)
            # Retrieve class name and value
            predicted_class = list(average[list(video_predictions.keys())[0]].keys())[max_idx]
            confidence = values[max_idx]
            print(predicted_class, confidence)
            with open(os.path.join(self.config['OUTPUT_FOLDER'], "DirectPrediction",
                                   f"Predictions_{list(video_predictions.keys())[0][8:-4]}.json"), "w") as f:
                video_predictions["checked"] = False
                json.dump(video_predictions, f, indent=2)
            event.clear()
            video_predictions = {}
