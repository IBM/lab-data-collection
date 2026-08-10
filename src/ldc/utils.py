import os
from datetime import datetime, timedelta

import cv2
import numpy as np

from ldc.gui import show_popup


LAST_FRAMES = [0] * 20
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
OUTPUT_FOLDER = "UPDATE_ME/Videoannotation/Data/Video annotation/Annotated videos/"
FOURCC = cv2.VideoWriter_fourcc(*"avc1")
FRAME_RATE = 10
SEC_TO_STOP = 15


###################################################################################################### For chemotion.py
class Selection:
    """
    Definition of Selection class for handling reactions in the data collector
    """
    def __init__(self, rect, ids, status):
        self.rect = rect
        self.id = ids
        self.status = status
        self.is_recording = False
        self.confirmed = False
        self.avg_B = None
        self.avg_G = None
        self.avg_R = None
        self.brightness = None
        self.avg_tmp = None


def quill_to_plain_text(quill_json):
    """
    Function to convert quill_json text in to plain text
    """
    plain_text = ""
    for op in quill_json.get("ops", []):
        if "insert" in op:
            plain_text += op["insert"]
    return plain_text


def action_append(root, item, selections, selected_item, instance, variables):
    """
    Function to append text in the description of the ELN in the form:
    '[%d/%m/%Y-%H:%M:%S] Action:'
    """
    if variables[selected_item] is not None:
        timestamp = datetime.now().strftime("[%d/%m/%Y-%H:%M:%S]")
        rea = instance.get_reaction(selections[variables[selected_item]].id)
        delta = rea.properties['description']
        new_line = f'\n{timestamp}: {item}'
        delta['ops'].append({"insert": new_line})
        rea.save()
        show_popup(root, f'Adding {item}!')


def list_objects_in_folder(folder_path, subfolder_name, int_var, variables,
                           nmr_list_lengths, selections, root, instance, selected_item):
    """
    Check the number of folder in the NMR folder if a new analysis is added it adds the text to the description
    automatically
    """
    print(subfolder_name, int_var[nmr_list_lengths])
    try:
        objects = os.listdir(folder_path)
        if isinstance(objects, str):
            return objects  # Return the error message if any
        if len(int_var[nmr_list_lengths]) == 0:
            print('first loop')
            for obj in objects:
                obj_path = os.path.join(folder_path, obj)
                folders = os.listdir(obj_path)
                int_var[nmr_list_lengths][obj] = len(folders)
            list_objects_in_folder(folder_path=folder_path,
                                   subfolder_name=subfolder_name,
                                   int_var=int_var,
                                   nmr_list_lengths=nmr_list_lengths,
                                   selections=selections,
                                   root=root,
                                   instance=instance,
                                   selected_item=selected_item,
                                   variables=variables)

        elif len(objects) == len(int_var[nmr_list_lengths]):
            print(1, 'same number of NMR')
            for obj in objects:
                obj_path = os.path.join(folder_path, obj)
                folders = os.listdir(obj_path)
                if int_var[nmr_list_lengths][obj] != len(folders):
                    print('new analysis on the same NMR')
                    items = [os.path.join(obj_path, item) for item in os.listdir(obj_path)]
                    most_recent = max(items, key=os.path.getmtime)
                    nmr_name = most_recent.removeprefix(os.path.join(folder_path, obj))[1:]
                    print(nmr_name, subfolder_name)
                    # for folder in folders:
                    if subfolder_name in nmr_name:
                        int_var[nmr_list_lengths][obj] = len(folders)
                        print(f'action added in {nmr_name}')
                        action_append(root=root,
                                      instance=instance,
                                      item=f'NMR analysis ({nmr_name}):',
                                      selected_item=selected_item,
                                      selections=selections,
                                      variables=variables)
        elif len(objects) != len(int_var[nmr_list_lengths]):
            print('new nmr')
            for obj in objects:
                if obj not in int_var[nmr_list_lengths].keys():
                    obj_path = os.path.join(folder_path, obj)
                    folders = os.listdir(obj_path)
                    items = [os.path.join(obj_path, item) for item in os.listdir(obj_path)]
                    most_recent = max(items, key=os.path.getmtime)
                    nmr_name = most_recent.removeprefix(os.path.join(folder_path, obj))[1:]
                    print(nmr_name, subfolder_name)
                    # for folder in folders:
                    if subfolder_name in nmr_name:
                        int_var[nmr_list_lengths][obj] = len(folders)
                        action_append(root=root,
                                      instance=instance,
                                      item=f'NMR analysis ({nmr_name}):',
                                      selected_item=selected_item,
                                      selections=selections,
                                      variables=variables)

        return f"Subfolder {subfolder_name} not found in {folder_path}"
    except FileNotFoundError:
        return f"The folder {folder_path} does not exist."
    except PermissionError:
        return f"Permission denied for accessing the folder {folder_path}."


################################################################################################# For fixed_cameras.py
def note_extractor(data):
    """
    Takes a dictionary type variable as input and gives as output a dictionary
    containing a string with the update time and the reaction name as key and a dictionary containing
    the reaction data as output.
    """
    reaction_data = {"description": ""}
    for i in data["description"]["ops"]:
        reaction_data["description"] += i["insert"] + " \n"
    return reaction_data


def LightMotion_sensor(frame, sensor, variables, thread_start, rec_off, one, two, rec, root, fl,  gr):
    global LAST_FRAMES
    Lum = "%6.0f" % sensor.get_currentValue()
    variables[gr] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variables[gr] = cv2.GaussianBlur(variables[gr], (15, 15), 0)
    if variables[fl] is None:
        variables[fl] = variables[gr]
    frameDelta = cv2.absdiff(variables[fl], variables[gr])
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    LAST_FRAMES.pop(0)
    LAST_FRAMES.append(thresh.sum())
    if int(Lum) > 10000:
        if np.count_nonzero(LAST_FRAMES) > 4:
            if variables[rec] is False:
                start_record(variables=variables,
                             rec=rec,
                             root=root,
                             one=one,
                             two=two)
                thread_start.set()
            variables[rec_off] = datetime.now() + timedelta(seconds=SEC_TO_STOP)
        if variables[rec_off] is not None:
            if datetime.now() > variables[rec_off] and variables[rec] is True:
                stop_recording(one=one,
                               two=two,
                               variables=variables,
                               rec=rec,
                               root=root)
                thread_start.clear()
    elif variables[rec] is True and int(Lum) <= 10000:
        stop_recording(one=one,
                       two=two,
                       variables=variables,
                       rec=rec,
                       root=root)
        thread_start.clear()
    else:
        pass


# Used in LightMotion_sensor
def start_record(variables, rec, root, one, two):
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if os.path.isdir(OUTPUT_FOLDER + f"/{current_datetime[0:10]}"):
        output_filename = os.path.join(
            OUTPUT_FOLDER, f"{current_datetime[0:10]}", f"Cam1_{current_datetime}.mp4"
        )
        output_filename1 = os.path.join(
            OUTPUT_FOLDER, f"{current_datetime[0:10]}", f"Cam2_{current_datetime}.mp4"
        )
        variables[one] = cv2.VideoWriter(
            output_filename, FOURCC, 9.87, (FRAME_WIDTH, FRAME_HEIGHT)
        )
        variables[two] = cv2.VideoWriter(
            output_filename1, FOURCC, 9.87, (FRAME_WIDTH, FRAME_HEIGHT)
        )
        if root is not False:
            show_popup(root=root, string=f"""Recording\n + Cam1_{current_datetime}.mp4 +
             \nand\n + Cam2_{current_datetime}.mp4 + \nstarted!""")
            variables[rec] = True
    else:
        os.mkdir(OUTPUT_FOLDER + f"/{current_datetime[0:10]}")
        start_record(variables=variables,
                     rec=rec,
                     root=root,
                     one=one,
                     two=two)


# Used in LightMotion_sensor
def stop_recording(one, two, variables, rec, root):
    variables[one].release()
    variables[two].release()
    variables[rec] = False
    show_popup(root, "Recordings stopped and saved!")



