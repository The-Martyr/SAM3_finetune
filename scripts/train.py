"""Entry point for torchrun, including multi-node launches."""
import os
from pathlib import Path
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam3.train.utils.train_utils import register_omegaconf_resolvers

root = Path(__file__).resolve().parents[1]
for name in ('SAM3_SOURCE', 'SAM3_CHECKPOINT', 'SAM3_DATA_ROOT', 'SAM3_ANNOTATION_FILE', 'SAM3_RUN_DIR'):
    if not os.environ.get(name):
        raise SystemExit('Required environment variable: ' + name)
for name in ('SAM3_SOURCE', 'SAM3_CHECKPOINT', 'SAM3_DATA_ROOT', 'SAM3_ANNOTATION_FILE'):
    if not Path(os.environ[name]).exists():
        raise SystemExit('Missing path: ' + name)
register_omegaconf_resolvers()
initialize_config_module('sam3.train', version_base='1.2')
cfg = compose(config_name='configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml')
OmegaConf.set_struct(cfg, False)
custom = OmegaConf.load(root / 'configs/train.yaml')
cfg = OmegaConf.merge(cfg, custom)
cfg.trainer.data.train = custom.trainer.data.train
cfg.trainer.loss = custom.trainer.loss
cfg.trainer.meters = {}
cfg.roboflow_train.supercategory = 'custom'
world = int(os.environ.get('WORLD_SIZE', '1'))
local_world = int(os.environ.get('LOCAL_WORLD_SIZE', str(world)))
cfg.launcher.num_nodes = world // local_world
cfg.launcher.gpus_per_node = local_world
cfg.submitit.use_cluster = False
if os.environ.get('SAM3_VALIDATE_CONFIG') == '1':
    print(OmegaConf.to_yaml(cfg, resolve=True))
else:
    instantiate(cfg.trainer, _recursive_=False).run()
