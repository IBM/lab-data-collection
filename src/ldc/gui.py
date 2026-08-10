import customtkinter as tk
from yoctopuce.yocto_api import YAPI
import threading


################################################################################################################ General
def show_popup(root, string: str):
    """
    Opens a popup with Title 'Warning' amd containing the text specified in the root specified
    """
    popup = tk.CTkToplevel(root)
    popup.title("Warning")
    # popup.iconbitmap('AnnotatorLogo1.ico')
    popup.geometry("200x100")

    label = tk.CTkLabel(popup, text=string)
    label.pack(pady=10)

    # Destroy the popup window after 1000 milliseconds (1 second)
    popup.after(1000, popup.destroy)


####################################################################################################### For chemotion.py
def ask_yes_no(message="Are you sure?"):
    result = {"value": None}

    def on_yes():
        result["value"] = True
        popup.destroy()

    def on_no():
        result["value"] = False
        popup.destroy()

    popup = tk.CTkToplevel()
    popup.title("Confirmation")
    popup.grab_set()  # Make it modal

    # Row 1: Label
    label = tk.CTkLabel(popup, text=message)
    label.grid(row=0, column=0, columnspan=2, padx=20, pady=10)

    # Row 2: Buttons
    btn_yes = tk.CTkButton(popup, text="Yes", width=10, command=on_yes)
    btn_no = tk.CTkButton(popup, text="No", width=10, command=on_no)

    btn_yes.grid(row=1, column=0, padx=10, pady=10)
    btn_no.grid(row=1, column=1, padx=10, pady=10)

    popup.wait_window()  # Wait until popup is closed
    return result["value"]

def show_longpopup(root, name: str, list1, desc):
    """
    Opens the popup with the reaction info. Used with right-click on the reaction listbox
    """
    popup1 = tk.CTkToplevel(root)
    popup1.title(f"Data {name}")
    i = None
    for i, row in enumerate(list1):
        for j, value in enumerate(row):
            label = tk.CTkLabel(
                popup1, text=value, padx=10, pady=5, wraplength=200)
            label.grid(row=i, column=j, sticky="nsew")
        # Make columns expand proportionally
    for j in range(len(list1[0])):
        popup1.grid_columnconfigure(j, weight=1)
    if len(desc) != 1:
        text_area = tk.CTkTextbox(popup1, width=300, height=150)
        text_area.insert(tk.INSERT, desc)
        text_area.configure(state="disabled")
        text_area.grid(row=i + 1, column=0, columnspan=2)

    buttonk = tk.CTkButton(popup1, text="Close", command=popup1.destroy)
    buttonk.grid(column=1, row=(len(list1) + 1), columnspan=2)


def show_react_var(variables, selected_item, selections, radio1, radio2):
    """
    If the region of interest of a reaction is defined, show the radiobuttons of the additional variables
    """
    if selections[variables[selected_item]].confirmed:  # If checkbox is checked (True)
        radio1.grid(row=5, column=3, sticky="w")
        radio2.grid(row=5, column=4, sticky="w")
    else:  # If checkbox is unchecked (False)
        radio1.grid_remove()
        radio2.grid_remove()


################################################################################################### For fixed_cameras.py
def close_window(root):
    """
    Close window of the specified root
    """

    YAPI.FreeAPI()
    root.destroy()
    root.quit()



