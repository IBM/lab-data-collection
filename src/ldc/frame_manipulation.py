from datetime import datetime

import cv2
import numpy as np

W_MIN = 100  # Define a maximum area for scaling
COLOR_INACTIVE = (255, 0, 0)  # [B, G, R]
COLOR_ACTIVE = (0, 0, 255)


################################################################################################## For fixed_cameras.py
def process_fixedframe(frame, n):
    """
    Function to add the text on the frame, returns the modified frame
    """
    timestamp = datetime.now().strftime("%H-%M-%S")
    cv2.putText(
        frame,
        f"Fixed Cam {n}: {timestamp}",
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    return frame


####################################################################################################### For chemotion.py
def process_frame(frame, variables, selected_item, selections):
    """
    Function to add the text adn the ROIs on the frame and saves the average color, returns the modified frame
    """
    for name, selection in selections.items():
        if name == variables[selected_item]:
            font_scale = 0.4
            w = (selection.rect[2] - selection.rect[0])
            if w < W_MIN:
                font_scale *= (w / W_MIN)
            if selection.is_recording is True and selection.confirmed is True:
                cv2.rectangle(frame,
                              (selection.rect[0] - 2, selection.rect[1] - 2),
                              (selection.rect[2] + 1, selection.rect[3] + 1),
                              COLOR_ACTIVE, 2)
                cv2.putText(frame, name, (selection.rect[0] + 5, selection.rect[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
                color = frame[
                        (selection.rect[3] * 7 + selection.rect[1]) // 8 - 10:(selection.rect[3] * 7 + selection.rect[
                            1]) // 8 + 10,
                        (selection.rect[2] + selection.rect[0]) // 2 - 10:(selection.rect[2] + selection.rect[0]) // 2 + 10]
                avg_color = color.mean(axis=(0, 1)) / 255
                selection.avg_B = avg_color[0]
                selection.avg_G = avg_color[1]
                selection.avg_R = avg_color[2]
                selection.brightness = np.mean(avg_color)
            elif selection.is_recording is False and name == variables[selected_item]:
                cv2.rectangle(frame,
                              (selection.rect[0] - 2, selection.rect[1] - 2),
                              (selection.rect[2] + 1, selection.rect[3] + 1),
                              COLOR_INACTIVE, 2)
                cv2.putText(frame, name, (selection.rect[0] + 5, selection.rect[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
                cv2.rectangle(frame, ((selection.rect[2] + selection.rect[0]) // 2 - 10,
                                      (selection.rect[3] * 7 + selection.rect[1]) // 8 - 10),
                              ((selection.rect[2] + selection.rect[0]) // 2 + 10,
                               (selection.rect[3] * 7 + selection.rect[1]) // 8 + 10), (0, 0, 0), 1)
            elif selection.is_recording is False:
                pass
    return frame


def process_tframe(tframe, selections, variables, selected_item):
    """
    Manipulation of the thermoframe into a normalized 640x480 frame and saves the average temp,
    returns the modified frame
    """
    tframe = np.reshape(tframe[0], (2, 192, 256, 2))  # reshape the raw frame
    raw = tframe[1, :, :, :].astype(np.intc)  # select only the lower image part
    raw = (raw[:, :, 1] << 8) + raw[:, :, 0]  # assemble the 16bit word
    temp = np.round(raw / 64.3 - 273.2, 1)  # convert to Celsius scale
    resized_temp = cv2.resize(temp, (640, 480),
                              interpolation=cv2.INTER_LINEAR)  # adjust to the size of the 'frame'
    tcal = resized_temp[70: 450, 75: 575]  # calibration cut: take the temperature frames that overlap with the 'frame'
    tcal = cv2.resize(tcal, (640, 480),
                      interpolation=cv2.INTER_LINEAR)  # resize the calibrated to the same size of the 'frame'
    for name, selection in selections.items():
        if selection.is_recording and selection.confirmed is True:
            temperatures = tcal[
                    (selection.rect[3] * 7 + selection.rect[1]) // 8 - 10:
                    (selection.rect[3] * 7 + selection.rect[1]) // 8 + 10,
                    (selection.rect[2] + selection.rect[0]) // 2 - 10:
                    (selection.rect[2] + selection.rect[0]) // 2 + 10]
            avg_temp = temperatures.mean(axis=(0, 1))
            selection.avg_tmp = round(avg_temp, 1)
    normalized_temp_8bit = cv2.normalize(tcal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colormap_temp = cv2.applyColorMap(normalized_temp_8bit, cv2.COLORMAP_PLASMA)

    for name, selection in selections.items():
        if name == variables[selected_item]:
            font_scale = 0.4
            w = (selection.rect[2] - selection.rect[0])
            if w < W_MIN:
                font_scale *= (w / W_MIN)
            if selection.is_recording is True and selection.confirmed is True:
                cv2.rectangle(colormap_temp,
                              (selection.rect[0] - 2, selection.rect[1] - 2),
                              (selection.rect[2] + 1, selection.rect[3] + 1),
                              COLOR_ACTIVE, 2)
                cv2.putText(colormap_temp, name, (selection.rect[0] + 5, selection.rect[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
            elif selection.is_recording is False and name == variables[selected_item]:
                cv2.rectangle(colormap_temp,
                              (selection.rect[0] - 2, selection.rect[1] - 2),
                              (selection.rect[2] + 1, selection.rect[3] + 1),
                              COLOR_INACTIVE, 2)
                cv2.putText(colormap_temp, name, (selection.rect[0] + 5, selection.rect[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
                cv2.rectangle(colormap_temp, ((selection.rect[2] + selection.rect[0]) // 2 - 10,
                                      (selection.rect[3] * 7 + selection.rect[1]) // 8 - 10),
                              ((selection.rect[2] + selection.rect[0]) // 2 + 10,
                               (selection.rect[3] * 7 + selection.rect[1]) // 8 + 10), (0, 0, 0), 1)
            elif selection.is_recording is False:
                pass
    return colormap_temp
