from re2g.datasets.v1 import RerankDataModule

PRETRAINED_MODEL_NAME = "monologg/koelectra-base-v3-discriminator"


def test_rerank_datamodule():
    rerank_datamodule = RerankDataModule(
        pretrained_model_name_or_path=PRETRAINED_MODEL_NAME,
        batch_size=8,
        bm25_k=4,
        dpr_k=4,
        seed=69,
    )
    val_dataloader = rerank_datamodule.val_dataloader()

    batch = next(iter(val_dataloader))

    assert batch["input_ids"].shape == (8, 9, 512)
    assert batch["attention_mask"].shape == (8, 9, 512)
    assert batch["token_type_ids"].shape == (8, 9, 512)
    assert batch["labels"].shape == (8, 9)
