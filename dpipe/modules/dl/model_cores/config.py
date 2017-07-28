from dpipe.modules.dl.model_cores.unet import UNet2D, UResNet2D
from .enet import ENet2D
from .deepmedic_orig import DeepMedicOrig
from .unet_like import UnetLike

model_core_name2model_core = {
    'deepmedic_orig': DeepMedicOrig,
    'enet2d': ENet2D,
    'unet2d': UNet2D,
    'uresnet2d': UResNet2D,
    'unetlike3d': UnetLike,
}
