from hydra.utils import instantiate
from omegaconf import OmegaConf

from aiaccel.util import aiaccel


def main(p):
    parameter_config = OmegaConf.create(p)
    trainer_config = OmegaConf.load("./trainer_config.yaml")
    nas_config = OmegaConf.load("./nas_config.yaml")
    trainer = instantiate(trainer_config.trainer, _partial_=True)
    t = trainer(nas_config, parameter_config)
    t.train()
    t.search()
    if nas_config.trainer.strategy == "ddp_find_unused_parameters_true" and nas_config.environment.gpus > 1:
        t.save()
    # trainer.retrain()

    if nas_config.trainer.strategy == "ddp_find_unused_parameters_true" and nas_config.environment.gpus > 1:
        if t.is_global_zero:
            valid_acc = t.get_valid_acc()
            print(valid_acc)
            return valid_acc
    else:
        valid_acc = t.get_valid_acc()
        print(valid_acc)
        return valid_acc


if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)
