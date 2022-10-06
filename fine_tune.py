import nemo.collections.asr as nemo_asr
from ruamel.yaml import YAML
import pytorch_lightning as pl
from omegaconf import DictConfig
import copy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os


def train():

    N_GPUS = 1
    N_DEVICES = 1
    EPOCHS = 100
    
    config_path = "./configs/quartznet5x5.yaml"
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)

    params['model']['train_ds']['manifest_filepath'] = "./data/train_manifest.jsonl"
    params['model']['validation_ds']['manifest_filepath'] = "./data/val_manifest.jsonl"

    params['model']['train_ds']['batch_size'] = 32 * N_GPUS * N_DEVICES

    model_to_load = "./pretrained_turkish_model/epoch-99.ckpt"


    first_asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']))
    first_asr_model = first_asr_model.load_from_checkpoint(model_to_load)


    first_asr_model.change_vocabulary(
        new_vocabulary=[" ", "a", "b", "c", "ç", "d", "e", "f", "g", "ğ", "h", "ı", "i", "j", "k", "l", "m",
                        "n", "o", "ö", "p", "q", "r", "s", "ş", "t", "u", "ü", "v", "w", "x", "y", "z", "'"])


    new_opt = copy.deepcopy(params['model']['optim'])

    new_opt['lr'] = 0.001
    # Point to the data we'll use for fine-tuning as the training set
    first_asr_model.setup_training_data(train_data_config=params['model']['train_ds'])
    # Point to the new validation data for fine-tuning
    first_asr_model.setup_validation_data(val_data_config=params['model']['validation_ds'])
    # assign optimizer config
    first_asr_model.setup_optimization(optim_config=DictConfig(new_opt))

    wandb_logger = WandbLogger(name="Quartznet5x5", project="TURKISH_FINETUNING")

    # used for saving models
    save_path = os.path.join(os.getcwd(),"TURKISH_FINETUNING" + "_" + "Quartznet5x5_models")
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        save_top_k= -1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix='',
        period=1
    )
    # DDP for multi gpu
    trainer = pl.Trainer(gpus=N_GPUS, accelerator='ddp',num_nodes=N_DEVICES,
                         max_epochs=EPOCHS, amp_level='O1', precision=16,
                         logger=wandb_logger, log_every_n_steps=150,
                         val_check_interval=1.0, checkpoint_callback=checkpoint_callback)

    first_asr_model.set_trainer(trainer)

    trainer.fit(first_asr_model)


if __name__ == '__main__':
    train()