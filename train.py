from dotenv import load_dotenv

from re2g.models.rerank import Rerank

try:
    load_dotenv()
except:
    print("Failed to load .env files..")

import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import WandbLogger
import wandb
from re2g.configs import settings
from re2g.datasets.v1 import DprDataModule
from re2g.datasets.v1 import RerankDataModule
from re2g.models.dpr import DPR

PROJECT = "re2g"

# Models
MODEL_NAME = settings.pretrained_model_name_or_path

NUM_QUERY_TRAINABLE_LAYERS = settings.context_num_trainable_layers

NUM_CONTEXT_TRAINABLE_LAYERS = settings.query_num_trainable_layers

# Optimizers
OPTIMIZER_LEARNING_RATE = settings.optimizer_learning_rate

OPTIMIZER_WEIGHT_DECAY = settings.optimizer_weight_decay

# Trainers
TRAINER_STRATEGY = settings.trainer_strategy

TRAINER_PRECISION = settings.trainer_precision

TRAINER_LIMIT_TRAIN_BATCHES = settings.trainer_limit_train_batches

TRAINER_LIMIT_VAL_BATCHES = settings.trainer_limit_val_batches

TRAINER_LIMIT_TEST_BATCHES = settings.trainer_limit_test_batches

TRAINER_MAX_EPOCHS = settings.trainer_max_epochs

# DataModules
DATAMODULE_BATCH_SIZE = settings.datamodule_batch_size

DATAMODULE_DPR_BM25_K = settings.datamodule_dpr_bm25_k

DATAMODULE_RERANK_BM25_K = settings.datamodule_rerank_bm25_k

# Checkpoints
CHECKPOINT_DIRPATH = settings.checkpoint_dirpath

CHECKPOINT_MONITOR = settings.checkpoint_monitor

CHECKPOINT_MODE = settings.checkpoint_mode

CHECKPOINT_EVERY_N_TRAIN_STEPS = settings.checkpoint_every_n_train_steps

CKPT_PATH = settings.ckpt_path or None

TRAINING_MODEL = settings.training_model

RERANK_LOSS_TYPE = settings.rerank_loss_type


def train_dpr():

    dpr = DPR(
        pretrained_model_name_or_path=MODEL_NAME,
        num_query_trainable_layers=NUM_QUERY_TRAINABLE_LAYERS,
        num_context_trainable_layers=NUM_CONTEXT_TRAINABLE_LAYERS,
        learning_rate=OPTIMIZER_LEARNING_RATE,
        weight_decay=OPTIMIZER_WEIGHT_DECAY,
    )

    wandb.init(project=PROJECT, config=dpr.hparams)
    wandb.watch(dpr, log="all", log_freq=1)

    logger = WandbLogger(log_model="all")

    dpr_datamodule = DprDataModule(
        pretrained_model_name_or_path=MODEL_NAME,
        batch_size=DATAMODULE_BATCH_SIZE,
        bm25_k=DATAMODULE_DPR_BM25_K,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=CHECKPOINT_DIRPATH,
            monitor=CHECKPOINT_MONITOR,
            mode=CHECKPOINT_MODE,
            every_n_train_steps=CHECKPOINT_EVERY_N_TRAIN_STEPS,
        ),
        ModelSummary(max_depth=-1),
    ]

    trainer = L.Trainer(
        limit_train_batches=TRAINER_LIMIT_TRAIN_BATCHES,
        limit_val_batches=TRAINER_LIMIT_VAL_BATCHES,
        limit_test_batches=TRAINER_LIMIT_TEST_BATCHES,
        max_epochs=TRAINER_MAX_EPOCHS,
        strategy=TRAINER_STRATEGY,
        precision=TRAINER_PRECISION,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model=dpr, datamodule=dpr_datamodule, ckpt_path=CKPT_PATH)


def train_rerank():

    rerank = Rerank(
        pretrained_model_name_or_path=MODEL_NAME,
        num_trainable_layers=NUM_QUERY_TRAINABLE_LAYERS,
        learning_rate=OPTIMIZER_LEARNING_RATE,
        weight_decay=OPTIMIZER_WEIGHT_DECAY,
        loss_type=RERANK_LOSS_TYPE,
    )

    wandb.init(project=PROJECT, config=rerank.hparams)
    wandb.watch(rerank, log="all", log_freq=1)

    logger = WandbLogger(log_model="all")

    rerank_datamodule = RerankDataModule(
        pretrained_model_name_or_path=MODEL_NAME,
        batch_size=DATAMODULE_BATCH_SIZE,
        bm25_k=DATAMODULE_DPR_BM25_K,
        dpr_k=DATAMODULE_RERANK_BM25_K,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=CHECKPOINT_DIRPATH,
            monitor=CHECKPOINT_MONITOR,
            mode=CHECKPOINT_MODE,
            every_n_train_steps=CHECKPOINT_EVERY_N_TRAIN_STEPS,
        ),
        ModelSummary(max_depth=-1),
    ]

    trainer = L.Trainer(
        limit_train_batches=TRAINER_LIMIT_TRAIN_BATCHES,
        limit_val_batches=TRAINER_LIMIT_VAL_BATCHES,
        limit_test_batches=TRAINER_LIMIT_TEST_BATCHES,
        max_epochs=TRAINER_MAX_EPOCHS,
        strategy=TRAINER_STRATEGY,
        precision=TRAINER_PRECISION,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model=rerank, datamodule=rerank_datamodule, ckpt_path=CKPT_PATH)


if __name__ == "__main__":
    for key, value in settings.dict().items():
        print(f"{key}: {value}")

    if TRAINING_MODEL == "dpr":
        train_dpr()
    elif TRAINING_MODEL == "rerank":
        train_rerank()
    else:
        raise ValueError(f"Invalid training model: {TRAINING_MODEL}")
