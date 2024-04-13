from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    pretrained_model_name_or_path: str = "monologg/koelectra-base-v3-discriminator"

    optimizer_learning_rate: float = 1e-5
    optimizer_weight_decay: float = 1e-2

    context_max_length: int = 512
    context_padding: str = "max_length"
    context_num_trainable_layers: int = 2

    query_max_length: int = 512
    query_padding: str = "max_length"
    query_num_trainable_layers: int = 2

    rerank_max_length: int = 512
    rerank_padding: str = "max_length"
    rerank_num_trainable_layers: int = 2

    datamodule_batch_size: int = 128
    datamodule_dpr_bm25_k: int = 64
    datamodule_rerank_bm25_k: int = 64
    datamodule_rerank_dpr_k: int = 64
    dataloader_num_workers: int = 0

    trainer_max_epochs: int = 100
    trainer_limit_val_batches: int | float = 1.0
    trainer_limit_train_batches: int | float = 1.0
    trainer_limit_test_batches: int | float = 1.0
    trainer_strategy: str = "auto"  # deepspeed_stage_1
    trainer_precision: int = 32

    checkpoint_dirpath: str = "checkpoints"
    checkpoint_monitor: str = "mrr"
    checkpoint_mode: str = "max"
    checkpoint_every_n_train_steps: int = 0
    ckpt_path: str | None = "last"

    training_model: str = "dpr"

    rerank_loss_type: str = "mpnl"


settings = Settings()
