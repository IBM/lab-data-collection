from ldc.chemotion import DATAiledELN
from ldc.Camera_selection import CameraSelector
from ldc.config_manager import load_config
from ldc.user_manager import user_exists, load_user
from ldc.setup_dialog import run_user

if not user_exists():
    run_user()
user = load_user()
config = load_config()
camera_selector = CameraSelector('MobileCamera', 'ThermoCamera')
DATAiledELN(config, user, camera=camera_selector.cameranum1,
            thermocamera=camera_selector.cameranum2).root.mainloop()
