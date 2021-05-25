import os
import torch
import argparse
import pickle

from nn.data.augmentation_preprocessing import get_train_aug_preproc, get_valid_aug_preproc
from nn.models.custom_efficientnet import CustomEfficientnet
from nn.models.classification_trainer import ClassificationTrainer
from nn.data import lock_deterministic
from nn.utils import get_param_from_config, object_from_dict

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
config_dir = os.path.normpath(os.path.join(SCRIPT_DIR, "../configs"))

parser = argparse.ArgumentParser()
parser.add_argument('--config', action="store", dest="config",
                    default=os.path.join(config_dir, "train_classifier.yaml"),
                    type=str)

args = parser.parse_args()


def main(train_config: dict):
    ROOT = train_config.root

    TRAIN_PATH = train_config.train_path
    VALID_PATH = train_config.valid_path
    # TEST_PATH = train_config.test_path

    DEVICE = torch.device(train_config.device)
    PREPROCESSING_FN = CustomEfficientnet.get_preprocess_fn()

    ROOT_TO_SAVE_MODEL = train_config.root_to_save_model

    lock_deterministic(train_config.seed)

    class_to_label = dict()
    dirs = os.listdir(os.path.join(ROOT, TRAIN_PATH))
    for i, d in enumerate(dirs):
        class_to_label[d] = i

    train_dataset = object_from_dict(train_config.dataset, root=ROOT, path=TRAIN_PATH,
                                     class_to_label=class_to_label,
                                     augmentation=get_train_aug_preproc(PREPROCESSING_FN))
    train_loader = train_dataset.get_dataloader(**train_config.train_dataloader, shuffle=True)

    valid_dataset = object_from_dict(train_config.dataset, root=ROOT, path=VALID_PATH,
                                     class_to_label=class_to_label,
                                     augmentation=get_valid_aug_preproc(PREPROCESSING_FN))
    valid_loader = valid_dataset.get_dataloader(**train_config.valid_dataloader, shuffle=False)

    # test_dataset = object_from_dict(train_config.dataset, root=ROOT, path=TEST_PATH,
    #                                 augmentation=get_valid_aug_preproc(PREPROCESSING_FN))

    # test_loader = valid_dataset.get_dataloader(**train_config.test_dataloader, shuffle=False)

    if not os.path.exists(os.path.join(train_config.root_to_save_model)):
        os.mkdir(os.path.join(train_config.root_to_save_model))

    model = object_from_dict(train_config.model).to(DEVICE)

    model_optimizer = object_from_dict(train_config.optimizer,
                                       params=filter(lambda x: x.requires_grad, model.get_cnn_parameters()))

    criterion = object_from_dict(train_config.criterion)

    scheduler = object_from_dict(train_config.scheduler, optimizer=model_optimizer)

    trainer = object_from_dict(
        train_config.trainer,
        model=model,
        criterion=criterion,
        optimizer=model_optimizer,
        scheduler=scheduler,
        trainloader=train_loader,
        validloader=valid_loader,
        root_to_save_model=ROOT_TO_SAVE_MODEL,
        device=DEVICE,
        config=train_config,
    )

    trainer.train_model()


if __name__ == "__main__":
    train_cfg = get_param_from_config(args.config)
    main(train_cfg)
