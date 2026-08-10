from ldc.fixed_cameras import GraphicUI
from ldc.Camera_selection import CameraSelector
from ldc.config_manager import config_exists, load_config
from ldc.setup_dialog import run_setup

if not config_exists():
    run_setup()
config = load_config()
camera_selector = CameraSelector('Top Camera', 'Bottom Camera')
GraphicUI(top_camera=camera_selector.cam1, bottom_camera=camera_selector.cam2, config=config).root.mainloop()
