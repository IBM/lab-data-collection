import customtkinter as ctk
from ldc.config_manager import save_config
from ldc.user_manager import save_user
from tkinter import filedialog
import re


def run_user():
    root = ctk.CTk()
    root.title("User Configuration")
    root.geometry("300x200")

    ctk.CTkLabel(root, text="Chemotion User").pack()
    user_entry = ctk.CTkEntry(root, width=250)
    user_entry.pack()

    ctk.CTkLabel(root, text="Chemotion Password").pack()
    pw_entry = ctk.CTkEntry(root, width=250)
    pw_entry.pack()

    def save():
        user = {
            "USER": user_entry.get(),
            "PASSWORD": pw_entry.get()
        }
        save_user(user)
        root.destroy()

    ctk.CTkButton(root, text="Save", command=save).pack()

    root.mainloop()

def run_setup():
    root = ctk.CTk()
    root.title("Initial Configuration")
    root.geometry("500x550")

    ctk.CTkLabel(root, text="Chemotion.json URL").pack()

    url_entry = ctk.CTkEntry(root, width=400)
    url_entry.pack()

    ctk.CTkLabel(root, text="Cookies").pack()
    cookies_entry = ctk.CTkTextbox(root, width=400, height=100, fg_color=("grey85", "grey20"))
    cookies_entry.pack()

    ctk.CTkLabel(root, text="Sensors URL").pack()
    sensor_entry = ctk.CTkEntry(root, width=400)
    sensor_entry.pack()


    def browse_file():
        file_path = filedialog.askopenfilename(
            title="Select a file",
            filetypes=[
                ("All files", "*.*"),
                ("JSON files", "*.json"),
                ("Images", "*.png *.jpg")
            ]
        )
        if file_path:
            entry_model.insert(index=0, string=file_path)

    # Label to show selected file
    ctk.CTkLabel(root, text="Model Selected").pack()
    entry_model = ctk.CTkEntry(root, width=400)
    entry_model.pack()
    # Button to open file browser
    root.btn = ctk.CTkButton(
        root,
        text="Browse file",
        command=browse_file
    )
    root.btn.pack(pady=20)

    def browse_folder():
        folder_path = filedialog.askdirectory()
        if folder_path:
            entry_folder.insert(0, folder_path)

    ctk.CTkLabel(root, text="Destination folder").pack()
    entry_folder = ctk.CTkEntry(root, width=400)
    entry_folder.pack()

    # Button to open file browser
    root.btn1 = ctk.CTkButton(
        root,
        text="Browse file",
        command=browse_folder
    )
    root.btn1.pack(pady=20)

    def save():
        cookie_dict = {}
        for cookie in cookies_entry.get("1.0","end-1c").replace("\n", "").split(";"):
            cookie = cookie.strip()
            if "=" in cookie:
                name, value = cookie.split("=", 1)
                cookie_dict[name] = value
        config = {
            "URL": re.sub(r"(&per_page=).*", r"\g<1>100", url_entry.get()),
            "COOKIES": cookie_dict,
            "MODEL_PATH": entry_model.get(),
            "URL SENSORS": sensor_entry.get(),
            "OUTPUT_FOLDER": entry_folder.get()
        }
        save_config(config)
        root.destroy()

    ctk.CTkButton(root, text="Save", command=save).pack()

    root.mainloop()
