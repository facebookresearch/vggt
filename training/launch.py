from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer


with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name="default")      # loads default.yaml

trainer = Trainer(**cfg)
trainer.run()
