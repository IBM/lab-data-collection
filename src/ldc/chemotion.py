import logging
import customtkinter as ctk
import sys
import threading
import os
import time
import csv
from functools import partial
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import json
from collections import defaultdict

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import matplotlib.pyplot as plt
from yoctopuce.yocto_api import YAPI, YRefParam
from yoctopuce.yocto_humidity import YHumidity
from yoctopuce.yocto_pressure import YPressure
from yoctopuce.yocto_temperature import YTemperature
from chemotion_api import Instance

from ldc.utils import action_append
from ldc.gui import show_react_var, ask_yes_no
from ldc.events import (on_select, refresh, start_reaction, stop_reaction, update_graph, show_frame,
                            on_right_click, on_right_mouse_drag, on_left_mouse_drag, show_action_list)
from ldc.frame_manipulation import process_frame, process_tframe


# Variables
FRAME_RATE = 20
FRAME_INTERVAL = 1.0 / FRAME_RATE
# List of actions to be annotated
ACTIONS_LIST = ['Addition of solids:', 'Stir:',
                'Inert atmosphere:', 'Addition of liquids:', 'Extraction:',
                'Filtration:', 'Reflux:', 'Concentrate:']
# Class definition
class DATAiledELN:

    def __init__(self, config, user, camera=0, thermocamera=2):
        # Class Variables
        self.config = config
        Path(os.path.join(self.config['OUTPUT_FOLDER'], "Reactions sensor data")).mkdir(parents=True, exist_ok=True)
        self.TempI = 0
        self.PresI = 0
        self.HumI = 0
        self.variables = {'cframe': None, 'ctframe': None, 'selected_item': None}
        self.selections = {}
        # Floating radiobuttons
        self.radio1 = 0
        self.radio2 = 0

        # Connection to APIs for ELN, sensor and camera
        self.instance = Instance(self.config['URL'][:25]).test_connection()
        errmsg = YRefParam()
        if YAPI.RegisterHub(self.config['URL SENSORS'], errmsg) != YAPI.SUCCESS:
            sys.exit("init error" + errmsg.value)
        self.humSensorI = YHumidity.FirstHumidity()
        self.pressSensorI = YPressure.FirstPressure()
        self.tempSensorI = YTemperature.FirstTemperature()
        self.cap = cv2.VideoCapture(int(camera))
        self.tcap = cv2.VideoCapture(int(thermocamera))
        self.tcap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        try:
            self.instance.login(user['USER'], user['PASSWORD'])
        except ConnectionError:
            logging.error(f"A connection to Chemotion ({self.instance.host_url}) cannot be established")

        rc = self.instance.get_root_collection(True)
        self.all_col = rc.get_collection('/')

        ######################################################################################### Tkinter GUI definition
        # Create the main window
        self.root = ctk.CTk()
        self.root.title("ELNScribe")

        # Variable that shows the item selected
        self.var = tk.StringVar()
        self.var.set(f'Selected: {self.variables["selected_item"]}')

        # Variable to store the selected option from radiobuttons
        self.selected_option = tk.StringVar(value="Temperature")

        # Label and listbox definition
        self.label1 = ctk.CTkLabel(self.root, text='Reactions List')
        self.label1.grid(column=0, row=0, columnspan=5)
        self.style = ttk.Style()
        self.style.configure("Treeview", rowheight=30)

        self.listbox1 = ttk.Treeview(self.root, show="tree", style='Treeview', height=6)

        # Define custom tags for colors in listbox
        self.listbox1.tag_configure("yellow", foreground="#FAB12F", font=("Arial", 20))
        self.listbox1.tag_configure("blue", foreground="blue", font=("Arial", 20))
        self.listbox1.tag_configure("black", foreground="black", font=("Arial", 20))

        # Scrollbar Listbox
        self.scrollbar1 = ctk.CTkScrollbar(self.root, orientation='vertical', width=20)
        self.scrollbar1.grid(column=2, row=1)
        self.listbox1.config(yscrollcommand=self.scrollbar1.set)
        self.scrollbar1.configure(command=self.listbox1.yview)

        # Set up right-click (on listbox)
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Select", command=partial(on_right_click,
                                                                      variables=self.variables,
                                                                      selected_item='selected_item',
                                                                      instance=self.instance,
                                                                      selections=self.selections,
                                                                      root=self.root,
                                                                      listbox1=self.listbox1))  # Right-click action
        self.listbox1.grid(column=0, row=1, columnspan=4, sticky="nsew")
        self.scrollbar1.grid(column=4, row=1, sticky="ns")

        # Define interactions with listbox
        self.listbox1.bind("<<TreeviewSelect>>", partial(on_select,
                                                         variables=self.variables,
                                                         selected_item='selected_item',
                                                         listbox1=self.listbox1,
                                                         var=self.var))
        self.listbox1.bind("<Button-3>", partial(on_right_click,
                                                 variables=self.variables,
                                                 selected_item='selected_item',
                                                 instance=self.instance,
                                                 selections=self.selections,
                                                 root=self.root,
                                                 listbox1=self.listbox1))

        # Label showing the selected item from the listbox
        self.label = ctk.CTkLabel(self.root,
                              textvariable=self.var)  # Selected rectangle variable label
        self.label.grid(column=0, row=2, columnspan=5)

        # Button for addition of actions
        self.button_show_actions = ctk.CTkButton(self.root,
                                                 text="Show Actions",
                                                 font=('Arial', 14),
                                                 # activebackground='blue',
                                                 command=partial(show_action_list,
                                                                 root=self.root,
                                                                 instance=self.instance,
                                                                 selections=self.selections,
                                                                 variables=self.variables,
                                                                 selected_item='selected_item',
                                                                 actions_list=ACTIONS_LIST))
        self.button_show_actions.grid(column=0, row=4, columnspan=2)

        # Start reaction button
        self.button = ctk.CTkButton(self.root,
                                    text="Start Reaction",
                                    font=('Arial', 14),
                                    command=partial(start_reaction,
                                                    root=self.root,
                                                    selections=self.selections,
                                                    variables=self.variables,
                                                    selected_item='selected_item',
                                                    all_col=self.all_col,
                                                    instance=self.instance,
                                                    listbox1=self.listbox1,
                                                    reaction_path=os.path.join(self.config['OUTPUT_FOLDER'], "Reactions sensor data")))  # "Start Recording" button
        self.button.grid(column=0, row=3, columnspan=2)

        # Stop reaction button
        self.button1 = ctk.CTkButton(self.root,
                                     text="Stop Reaction",
                                     font=('Arial', 14),
                                     command=partial(stop_reaction,
                                                     root=self.root,
                                                     selections=self.selections,
                                                     variables=self.variables,
                                                     selected_item='selected_item',
                                                     all_col=self.all_col,
                                                     instance=self.instance,
                                                     listbox1=self.listbox1))  # "Stop Recording" button
        self.button1.grid(column=2, row=3, columnspan=3)

        # Refresh listbox buttons
        self.button2 = ctk.CTkButton(self.root,
                                     text='Refresh',
                                     font=('Arial', 14),
                                     command=partial(refresh,
                                                     root=self.root,
                                                     listbox1=self.listbox1,
                                                     all_col=self.all_col,
                                                     variables=self.variables,
                                                     selected_item='selected_item',
                                                     selections=self.selections))  # "Close" program button
        self.button2.grid(column=0, row=7, columnspan=5)

        # Open cameras button
        self.button3 = ctk.CTkButton(self.root,
                                     text='Cameras',
                                     font=('Arial', 14),
                                     command=self.open_camera_popup)

        self.button3.grid(column=2, row=4, columnspan=3)

        # Menu to select what to show in the graph
        self.menu_frame = ctk.CTkFrame(self.root)
        self. menu_frame.grid(column=0, row=6, columnspan=5, sticky="nsew")

        # Figure and axis for the plot
        self.fig, self.ax = plt.subplots(figsize=(3, 2))

        # Canvas to display the plot in the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=0, row=6, columnspan=5, sticky="nsew")

        # Radiobuttons to select what to show in the graph
        self.temp_radio = tk.Radiobutton(self.root, text="r.t.",
                                         variable=self.selected_option,
                                         value="Temperature",
                                         command=partial(update_graph,
                                                         ax=self.ax,
                                                         selections=self.selections,
                                                         selected_option=self.selected_option,
                                                         canvas=self.canvas,
                                                         variables=self.variables,
                                                         selected_item='selected_item'))
        self.temp_radio.grid(column=0, row=5, sticky="w")

        self.pressure_radio = tk.Radiobutton(self.root,
                                             text="Pres",
                                             variable=self.selected_option,
                                             value="Pressure",
                                             command=partial(update_graph,
                                                             ax=self.ax,
                                                             selections=self.selections,
                                                             selected_option=self.selected_option,
                                                             canvas=self.canvas,
                                                             variables=self.variables,
                                                             selected_item='selected_item'))
        self.pressure_radio.grid(column=1, row=5, sticky="w")

        self.humidity_radio = tk.Radiobutton(self.root,
                                             text="Hum",
                                             variable=self.selected_option,
                                             value="Humidity",
                                             command=partial(update_graph,
                                                             ax=self.ax,
                                                             selections=self.selections,
                                                             selected_option=self.selected_option,
                                                             canvas=self.canvas,
                                                             variables=self.variables,
                                                             selected_item='selected_item'))
        self.humidity_radio.grid(column=2, row=5, sticky="w")

        self.radio1 = tk.Radiobutton(self.root,
                                     text="Color",
                                     variable=self.selected_option,
                                     value="avg_color",
                                     command=partial(update_graph,
                                                     ax=self.ax,
                                                     selections=self.selections,
                                                     selected_option=self.selected_option,
                                                     canvas=self.canvas,
                                                     variables=self.variables,
                                                     selected_item='selected_item'))

        self.radio2 = tk.Radiobutton(self.root,
                                     text="Temp", variable=self.selected_option,
                                     value="avg_temp",
                                     command=partial(update_graph,
                                                     ax=self.ax,
                                                     selections=self.selections,
                                                     selected_option=self.selected_option,
                                                     canvas=self.canvas,
                                                     variables=self.variables,
                                                     selected_item='selected_item'))

        # Update the graph whenever a reaction is selected or refreshed
        self.listbox1.bind("<<TreeviewSelect>>", lambda event: [on_select(event=event,
                                                                          listbox1=self.listbox1,
                                                                          var=self.var,
                                                                          variables=self.variables,
                                                                          selected_item='selected_item'),
                                                                update_graph(ax=self.ax,
                                                                             selections=self.selections,
                                                                             selected_option=self.selected_option,
                                                                             canvas=self.canvas,
                                                                             variables=self.variables,
                                                                             selected_item='selected_item'),
                                                                show_react_var(variables=self.variables,
                                                                               selected_item='selected_item',
                                                                               selections=self.selections,
                                                                               radio1=self.radio1,
                                                                               radio2=self.radio2)])
        time.sleep(5)
        # Thread for acquisition of frames from the cameras
        self.cameras = threading.Thread(target=self.cameras_acquisition, daemon=True)
        self.cameras.start()

        # Thread for data acquisition (from sensor)
        self.data = threading.Thread(target=self.data_acquisition, daemon=True)
        self.data.start()

        time.sleep(5)

        refresh(root=self.root,
                listbox1=self.listbox1,
                all_col=self.all_col,
                variables=self.variables,
                selected_item='selected_item',
                selections=self.selections)

        # Start the Tkinter main loop
        self.root.mainloop()

        self.data.join()
        self.cameras.join()

        # Release cameras and close windows when GUI is closed
        self.cap.release()
        self.tcap.release()
        cv2.destroyAllWindows()

    # Data acquisition loop (thread)
    def data_acquisition(self):
        # Variables for NMR analysis addition
        counter = 0
        # Start loop
        while True:
            start_time = time.time()    # Starting time
            if counter == 29:
                # Fetching values from sensor
                TempI = "%3.1f" % self.tempSensorI.get_currentValue()
                PresI = "%4.0f" % self.pressSensorI.get_currentValue()
                HumI = "%3.1f" % self.humSensorI.get_currentValue()

                selections = self.selections.copy()

                for name, selection in selections.items():     # Loops over the reactions
                    if selection.status == "Running":               # if the reaction is running
                        reaction_filename = os.path.join(os.path.join(self.config['OUTPUT_FOLDER'], "Reactions sensor data"), f"{name}.csv")     # Path of reaction csv
                        Fnew_data = [
                            TempI,
                            PresI,
                            HumI,
                            selection.avg_B,
                            selection.avg_G,
                            selection.avg_R,
                            selection.brightness,
                            selection.avg_tmp
                        ]                                             # List containing all the values to append
                        if os.path.exists(reaction_filename):           # Creation of csv and addition of values
                            with open(reaction_filename, "a", newline="") as reaction_file:
                                writer = csv.writer(reaction_file)
                                writer.writerow(Fnew_data)
                        # for name, selection in self.selections.items():
                        rea = self.instance.get_reaction(selection.id)              # Fetch reaction from relative id
                        # If no temperature is specified and the reaction is running
                        if rea.properties["temperature"]["userText"] == "" and rea.properties['status'] == "Running":
                            # reaction_name = rea.properties["name"]
                            reaction_filename = os.path.join(os.path.join(self.config['OUTPUT_FOLDER'],
                                                                          "Reactions sensor data"), f"{name}.csv")
                            df = pd.read_csv(reaction_filename)
                            try:
                                last_row = df.iloc[-1]['Temperature']
                                h = df.index[-1] // 60
                                m = df.index[-1] % 60
                                s = 0
                                # Add temperature value in the last row
                                rea.properties["temperature"].add_time_point(h, m, s, last_row)
                                rea.save()      # Add changes
                            except IndexError:
                                pass
                    # Update graphs after addition of the new datapoint
                    update_graph(ax=self.ax,
                                 selections=self.selections,
                                 selected_option=self.selected_option,
                                 canvas=self.canvas,
                                 variables=self.variables,
                                 selected_item='selected_item')
                counter = 0
            directory = Path(os.path.join(self.config['OUTPUT_FOLDER'], "DirectPrediction"))
            recent_files = list(directory.glob("*.json"))
            latest_file = max(recent_files, key=lambda f: f.stat().st_mtime)
            print(latest_file)
            if latest_file:
                with open(latest_file, "r") as f:
                    data = json.load(f)
                if not data["checked"]:
                    with open(latest_file, "w") as f:
                        data["checked"] = True
                        json.dump(data, f, indent=2)
                    prob_sums = defaultdict(float)

                    count = 0

                    # Loop through each video entry
                    for entry in data.values():
                        if type(entry) is not bool:
                            all_probs = entry["all_probabilities"]

                            for class_name, prob in all_probs.items():
                                prob_sums[class_name] += prob

                            count += 1
                    # Compute averages
                    avg_probs = {class_name: total / count
                                 for class_name, total in prob_sums.items()}
                    if avg_probs and isinstance(avg_probs, dict):
                        max_key = max(avg_probs, key=avg_probs.get)
                    else:
                        print("No valid probability dictionary found.")

                    response = ask_yes_no("Prediction: "+max_key)

                    if response:
                        action_append(self.root,
                                      max_key,
                                      self.selections,
                                      "selected_item",
                                      self.instance,
                                      self.variables)
                    else:
                        show_action_list(
                                         root=self.root,
                                         instance=self.instance,
                                         selections=self.selections,
                                         variables=self.variables,
                                         selected_item='selected_item',
                                         actions_list=ACTIONS_LIST)

                counter += 1
            elapsed_time = time.time() - start_time     # Check how long did it take to run the code
            time_to_wait = max(0, 2 - elapsed_time)    # loop in 2 seconds
            time.sleep(time_to_wait)

    # Thread camera and thermocamera acquisition
    def cameras_acquisition(self):
        while True:
            start_time = time.time()            # Starting time
            ret, frame = self.cap.read()        # Fetch frame from normal camera
            tret, tframe = self.tcap.read()     # Fetch thermoframe from thermocamera
            if ret and tret:
                # Process frame
                pframe = process_frame(frame=frame,
                                       variables=self.variables,
                                       selected_item='selected_item',
                                       selections=self.selections)
                # Process thermoframe
                ntframe = process_tframe(tframe=tframe,
                                         selections=self.selections,
                                         variables=self.variables,
                                         selected_item='selected_item')
                # Save frame and thermoframe as variable values that can be read by other threads.
                self.variables['cframe'] = pframe
                self.variables['ctframe'] = ntframe

            else:
                time.sleep(5)
                self.cameras_acquisition()
                # Wait the loop to repeat in the defined framerate
            time.sleep(max(0, FRAME_INTERVAL - (time.time() - start_time)))

    # Mobile camera interface
    def open_camera_popup(self):
        croot = ctk.CTkToplevel(self.root)              # Open toplevel window
        croot.title('Mobile cameras')
        calibration_confirm = {'calibrate': False,  # define variables
                               'destroy': False}

        # Function for activating the calibration mode
        def calibration():
            calibration_confirm['calibrate'] = True

        # Function for to confirm the position of the region of interest
        def confirmed():
            self.selections[self.variables['selected_item']].confirmed = True
            calibration_confirm['destroy'] = True
            show_react_var(variables=self.variables,
                           selected_item='selected_item',
                           selections=self.selections,
                           radio1=self.radio1,
                           radio2=self.radio2)

        # New window GUI widgets
        label = ctk.CTkLabel(croot, text=self.variables['selected_item'])  # Selected rectangle variable label
        label.grid(column=0, row=0, columnspan=3)

        panel = ctk.CTkLabel(croot)  # Video panel
        panel.grid(column=0, row=1, columnspan=3)

        button2 = ctk.CTkButton(croot, text='Confirm', command=confirmed)  # "Close" program button
        button2.grid(column=0, row=2)

        button3 = ctk.CTkButton(croot, text="Calibration", command=calibration)     # "Add reaction" rectangle button
        button3.grid(column=1, row=2)

        buttonk = ctk.CTkButton(croot, text="Close", command=croot.destroy)     # "Delete" rectangle button
        buttonk.grid(column=2, row=2)

        # Interactions with panel
        panel.bind("<B1-Motion>", partial(on_left_mouse_drag,
                                          selections=self.selections,
                                          variables=self.variables,
                                          selected_item='selected_item'))
        panel.bind("<B3-Motion>", partial(on_right_mouse_drag,
                                          selections=self.selections,
                                          variables=self.variables,
                                          selected_item='selected_item'))

        # (Recursive) Function to show the video in the panel
        show_frame(cframe='cframe',
                   ctframe='ctframe',
                   variables=self.variables,
                   panel=panel,
                   croot=croot,
                   calibration_on=calibration_confirm,
                   calibrate='calibrate',
                   destroy='destroy')
