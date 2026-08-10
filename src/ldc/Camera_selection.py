import cv2
import os
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from PIL import Image
from pathlib import Path


APP_DIR = Path(os.getenv("APPDATA")) / "LDC"
APP_DIR.mkdir(parents=True, exist_ok=True)

DIR_FILE = APP_DIR / "saved_dir.json"

class CameraSelector:
    def __init__(self, name_1, name_2):
        self.root = ctk.CTk()
        self.root.title("Camera Selection")
        self.cam1 = None
        self.cam2 = None
        self.cameranum1 = None
        self.cameranum2 = None

        self.camera_list = self.list_cameras()

        if not self.camera_list.keys():
            messagebox.showerror("No Cameras", "No available cameras found!")
            self.root.destroy()
            return

        ctk.CTkLabel(self.root, text="Select the cameras:").pack()

        self.camera_listbox = tk.Listbox(self.root, height=len(self.camera_list), selectmode=tk.SINGLE)
        self.camera_listbox.pack()

        for cam in self.camera_list.keys():
            self.camera_listbox.insert(tk.END, f"Camera {cam}")

        if self.camera_list:
            self.camera_listbox.selection_set(0)
            self.camera_listbox.activate(0)

        self.camera_listbox.bind("<<ListboxSelect>>", self.update_selection)

        self.selected_camera = ctk.StringVar(value="Selected Camera: None")
        ctk.CTkLabel(self.root, textvariable=self.selected_camera).pack()

        # Dropdowns for manual camera selection
        ctk.CTkLabel(self.root, text=f'{name_1}:').pack()
        self.camera_dropdown1 = ctk.CTkComboBox(self.root, values=[str(k) for k in self.camera_list.keys()])
        self.camera_dropdown1.pack()

        ctk.CTkLabel(self.root, text=f'{name_2}:').pack()
        self.camera_dropdown2 = ctk.CTkComboBox(self.root, values=[str(k) for k in self.camera_list.keys()])
        self.camera_dropdown2.pack()

        ctk.CTkButton(self.root, text="Save", command=self.save).pack()

        # Image panel
        self.panel = ctk.CTkLabel(self.root)
        self.panel.pack()

        self.update_preview(0)  # Show first camera preview by default
        self.root.mainloop()



    def list_cameras(self):
        """Detect available cameras."""
        available_cameras = {}
        for index in range(4):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras[len(available_cameras)] = [frame, cap]
        return available_cameras

    def update_selection(self, event):
        """Update the preview when a camera is selected."""
        selected_index = self.camera_listbox.curselection()
        if selected_index:
            cam_index = list(self.camera_list.keys())[selected_index[0]]
            self.selected_camera.set(f"Selected Camera: {cam_index}")
            self.update_preview(cam_index)

    def update_preview(self, cam_index):
        """Update the panel image with the selected camera preview."""
        if cam_index in self.camera_list.keys():
            frame = cv2.cvtColor(self.camera_list[cam_index][0], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(500, 400))
            self.panel.imgtk = imgtk
            self.panel.configure(image=imgtk, text="")

    def save(self):
        """Save the selected cameras."""

        self.cameranum1 = self.camera_dropdown1.get()
        self.cameranum2 = self.camera_dropdown2.get()

        self.cam1 = self.camera_list[int(self.cameranum1)][1]
        self.cam2 = self.camera_list[int(self.cameranum2)][1]

        if self.cam1 == self.cam2:
            messagebox.showerror(
                "Selected Cameras Error",
                "Select two different cameras"
            )
            return

        print(
            "Selected Cameras:",
            self.cameranum1,
            self.cameranum2
        )

        # release unused cameras safely
        selected = {
            int(self.cameranum1),
            int(self.cameranum2)
        }

        for camera_id, cam in self.camera_list.items():
            if camera_id not in selected:
                cam[1].release()

        # OPTIONAL: give Tk a clean exit frame
        self.root.after(0, self.root.destroy)



