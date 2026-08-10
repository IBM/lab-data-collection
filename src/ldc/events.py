from datetime import datetime
import customtkinter as ctk
import tkinter as tk
import os
import csv
from functools import partial

import pubchempy
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import cv2
import pandas as pd
from PIL import Image

from ldc.utils import quill_to_plain_text, Selection, action_append
from ldc.gui import show_popup, show_longpopup

import matplotlib.pyplot as plt

CORR = 0.15


###################################################################################################### For chemotion.py
def refresh(root, listbox1, all_col, variables, selected_item,  selections):
    """
    Updates the listbox with new reactions or removes finished reactions
    """
    show_popup(root, 'Refreshing...')
    listbox1.delete(*listbox1.get_children())
    for r in all_col.get_reactions()[:100]:
        if r.properties['status'] == 'Successful' or r.properties['status'] == 'Not Successful':
            pass
        elif r.properties['status'] == 'Running':
            listbox1.insert("", "end", text=r.properties['name'], tags=("yellow",))
            if r.properties['name'] not in selections.keys():
                selections[r.properties['name']] = Selection([50, 50, 150, 150], r.id, 'Running')
            elif selections[r.properties['name']].status != 'Running':
                selections[r.properties['name']].status = 'Running'
        elif r.properties['status'] == 'Done':
            listbox1.insert("", "end", text=r.properties['name'], tags=("blue",))
            if r.properties['name'] not in selections.keys():
                selections[r.properties['name']] = Selection([50, 50, 150, 150], r.id, 'Done')
            elif selections[r.properties['name']].status != 'Done':
                selections[r.properties['name']].status = 'Done'
        else:
            listbox1.insert("", "end", text=r.properties['name'], tags=("black",))
            if r.properties['name'] not in selections.keys():
                selections[r.properties['name']] = Selection([50, 50, 150, 150], r.id, 'Planned')
            elif selections[r.properties['name']].status != 'Planned':
                selections[r.properties['name']].status = 'Planned'
    if listbox1.get_children():
        listbox1.selection_set(listbox1.get_children()[0])
        variables[selected_item] = listbox1.item(listbox1.get_children()[0], "text")


def on_select(event, listbox1,  var, variables, selected_item):
    """
    Associates the reaction selected in the listbox with the selected item
    """
    try:
        variables[selected_item] = listbox1.item(listbox1.selection()[0], "text")
        var.set(f'Selected: {variables[selected_item]}')
    except IndexError:
        pass


def on_left_mouse_drag(event, selections, variables, selected_item):
    """
    On left click, changes the position of the bottom right corner of the ROI
    """
    if not selections[variables[selected_item]].is_recording:
        selections[variables[selected_item]].rect[2] = event.x
        selections[variables[selected_item]].rect[3] = event.y


def on_right_mouse_drag(event, variables, selected_item, selections):
    """
    On right click, changes the position of the whole ROI (translation)
    """
    if not selections[variables[selected_item]].is_recording:
        width = abs(selections[variables[selected_item]].rect[0] - selections[variables[selected_item]].rect[2])
        height = abs(selections[variables[selected_item]].rect[1] - selections[variables[selected_item]].rect[3])
        selections[variables[selected_item]].rect[0] = event.x - int(0.5*width)
        selections[variables[selected_item]].rect[1] = event.y - int(0.5*height)
        selections[variables[selected_item]].rect[2] = event.x + int(0.5*width)
        selections[variables[selected_item]].rect[3] = event.y + int(0.5*height)


def start_reaction(root, selections, variables, selected_item, all_col, instance, listbox1, reaction_path):
    """
    Changes the status of the selected reaction to Running,
    adds the starting time and creates the csv file appending the column names.
    Changes the is_recording variable to True and shows a popup for completion of the actions.
    Finally, refreshes the listbox to update the color of the entry
    """
    if variables[selected_item] is not None:
        timestamp = datetime.now()
        rea = instance.get_reaction(selections[variables[selected_item]].id)
        if rea.properties['status'] == '' or rea.properties['status'] == 'Planned':
            rea.properties['temperature']['data'] = []
            rea.properties['status'] = 'Running'
            rea.properties['timestamp_start'] = timestamp
            output_reaction_filename = os.path.join(reaction_path, f"{rea.properties['name']}.csv")
            reaction_data = [
                ["Temperature", "Pressure", "Humidity", "avg_B", 'avg_G', 'avg_R', 'Brightness', "Thermo"],
            ]
            with open(output_reaction_filename, "w", newline="") as reaction_file:
                writer = csv.writer(reaction_file)
                writer.writerows(reaction_data)
            rea.save()
            selections[variables[selected_item]].is_recording = True
            show_popup(root, f'Starting {variables[selected_item]}...')
        refresh(root=root,
                listbox1=listbox1,
                all_col=all_col,
                variables=variables,
                selected_item='selected_item',
                selections=selections)


def stop_reaction(root, selections, variables, selected_item, all_col, instance, listbox1):
    """
    Changes the status of the selected reaction to Done, adds the ending time,
    changes the is_recording variable to False and shows a popup for completion of the actions.
    Finally, refreshes the listbox to update the color of the entry
    """
    if variables[selected_item] is not None:
        timestamp = datetime.now()
        rea = instance.get_reaction(selections[variables[selected_item]].id)
        if rea.properties['status'] == 'Running':
            rea.properties['status'] = 'Done'
            rea.properties['timestamp_stop'] = timestamp
            if len(rea.properties['temperature']['data']) != 0:
                rea.properties['temperature']['data'].pop()
            # print(type(rea.properties['timestamp_start']))
            # print(rea.properties['timestamp_start'])
            # print(timestamp)
            rea.save()
            selections[variables[selected_item]].is_recording = False
            show_popup(root, f'Stopping {variables[selected_item]}...')
        refresh(root=root,
                listbox1=listbox1,
                all_col=all_col,
                variables=variables,
                selected_item='selected_item',
                selections=selections)


def on_right_click(event, variables, selected_item, instance, selections, root, listbox1):
    """
    Fetches all the components of the reactions and creates a table with the values that are required.
    Shows the table and the text in observations (procedure to follow) in a long_popup.
    """
    variables[selected_item] = listbox1.item(listbox1.selection()[0], "text")
    rea = instance.get_reaction(selections[variables[selected_item]].id)
    components = []
    for i in rea.properties['starting_materials']:
        components.append(i)
    for i in rea.properties['reactants']:
        components.append(i)
    for i in rea.properties['solvents']:
        components.append(i)
    desc = quill_to_plain_text(rea.properties['observation'])
    amounts = []
    value = None
    for j in components:
        pass
        if pubchempy.get_compounds(j.molecule['cano_smiles'], namespace='smiles')[0].iupac_name is None:
            mol = Chem.MolFromSmiles(j.molecule['cano_smiles'])
            name1 = Chem.rdMolDescriptors.CalcMolFormula(mol)
        else:
            name1 = pubchempy.get_compounds(j.molecule['cano_smiles'], namespace='smiles')[0].iupac_name
        if j.properties['target_amount']['unit'] == 'mol':
            if j.properties['molarity']['value'] == 0 and j.properties['density'] == 0:
                value = f"""{round(j.properties["target_amount"]["value"] * 1000
                             * Chem.rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(j.molecule['cano_smiles'])),
                             2)} mg"""
            elif j.properties['density'] != 0:
                value = f"""{round(j.properties['target_amount']['value'] *
                             Chem.rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(j.molecule['cano_smiles'])) /
                             j.properties['density'], 3)} mL"""
            elif j.properties['molarity']['value'] != 0:
                value = f"""{round((j.properties['target_amount']['value'] * 1000)
                             / j.properties['molarity']['value'], 3)} mL"""
        elif j.properties['target_amount']['unit'] == 'g':
            if j.properties['molarity']['value'] == 0 and j.properties['density'] == 0:
                value = f"""{round(j.properties['target_amount']['value']*1000, 2)} mg"""
            elif j.properties['density'] != 0:
                value = f"""{round(j.properties['target_amount']['value']/j.properties['density'], 3)} mL"""
            elif j.properties['molarity']['value'] != 0:
                value = f"""{round((j.properties['target_amount']['value']*1000 /
                             Chem.rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(j.molecule['cano_smiles']))) /
                             j.properties['molarity']['value'], 3)} mL"""
        elif j.properties['target_amount']['unit'] == 'l':
            value = f"""{round(j.properties['target_amount']['value']*1000, 3)} mL"""
        formatted_text = name1.replace(';', '\n')
        amounts.append((formatted_text, value))
    show_longpopup(root, variables[selected_item], amounts, desc)


def show_frame(panel, croot, variables, cframe, ctframe, calibration_on, calibrate, destroy):
    """
    Function to show the frames.
    """
    if calibration_on[destroy]:
        croot.destroy()
    else:
        if calibration_on[calibrate]:
            while True:
                comp = np.hstack((variables[cframe], variables[ctframe]))
                cv2.imshow('Calibration panel', comp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyWindow(f'Calibration panel')
                    calibration_on[calibrate] = False
                    break
            show_frame(panel, croot, variables, cframe, ctframe, calibration_on, calibrate, destroy)
        else:
            FRAME = cv2.cvtColor(variables[cframe], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(FRAME)
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            panel.imgtk = imgtk
            try:
                panel.configure(image=imgtk, text='')
                croot.update()
                panel.after(10, partial(show_frame, panel, croot, variables, cframe,
                                        ctframe, calibration_on, calibrate, destroy))
            except tk.TclError:
                pass


def update_graph(ax, selections, selected_option, canvas, variables, selected_item):
    """
    Takes the values from the csv and shows them in the canvas of the GUI
    """
    if (variables[selected_item] is not None and
            (selections[variables[selected_item]].status == 'Running' or
             selections[variables[selected_item]].status == 'Done')):
        # TODO: as variable
        if os.path.exists(f"UPDATE_ME/Videoannotation/Data/Feature extraction/Reactions data/{variables[selected_item]}.csv"):
            df = pd.read_csv(f"UPDATE_ME/Videoannotation/Data/Feature extraction/Reactions data/{variables[selected_item]}.csv")
            ax.clear()
            ax.set_position([0.05, 0.19, 0.8, 0.8])
            if selected_option.get() == "Temperature":
                ax.plot(df.index/120, df["Temperature"], linestyle='-', label='Room Temp.', color='pink')
                ax.legend(fontsize=8)
            elif selected_option.get() == "Pressure":
                ax.plot(df.index/120, df["Pressure"], linestyle='-', label='Atm. Pres.', color='green')
                ax.legend(fontsize=8)
            elif selected_option.get() == "Humidity":
                ax.plot(df.index/120, df["Humidity"], linestyle='-', label='Hum %')
                ax.legend(fontsize=8)
            elif selected_option.get() == "avg_color" and selections[variables[selected_item]].confirmed:
                for i in df.index:
                    color = (float(df.at[i, 'avg_R']) + CORR,
                             float(df.at[i, 'avg_G']) + CORR,
                             float(df.at[i, 'avg_B']) + CORR)
                    ax.scatter(i / 120, df.at[i, 'Brightness'], color=color, label='Reac. Color', s=20)
            elif selected_option.get() == "avg_temp" and selections[variables[selected_item]].confirmed:
                ax.plot(df.index / 120, df["Thermo"], linestyle='-', label='Reac. Temp.', color='red')
                ax.legend(fontsize=8)
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_xlabel('Time (h)', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=8)
        try:
            canvas.draw()
        except AttributeError:
            pass


def show_action_list(root, instance, selections, variables, selected_item, actions_list):
    """
    Shows a popup with the list of action and allow their selection and addition in the ELN.
    """
    popup = ctk.CTkToplevel(root)
    popup.title("Action List")

    action_listbox = tk.Listbox(popup, font='Arial 20')
    action_listbox.grid(row=0, column=0, columnspan=2)

    for action in actions_list:
        action_listbox.insert(tk.END, action)

    def add_selected_action(root, instance, selections, variables, selected_item):
        selected_action = action_listbox.get(action_listbox.curselection())
        action_append(root=root,
                      item=selected_action,
                      selections=selections,
                      variables=variables,
                      selected_item=selected_item,
                      instance=instance)
        popup.destroy()

    button_add = ctk.CTkButton(popup, text="Add", command=partial(add_selected_action,
                                                              root=root,
                                                              instance=instance,
                                                              selections=selections,
                                                              variables=variables,
                                                              selected_item=selected_item
                                                              ))
    button_add.grid(row=1, column=0)

    button_close = ctk.CTkButton(popup, text="Close", command=popup.destroy)
    button_close.grid(row=1, column=1)
