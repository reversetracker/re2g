from re2g.datasets.v1 import DprDataModule

PRETRAINED_MODEL_NAME = "monologg/koelectra-base-v3-discriminator"


def test_dpr_data_module():
    data_module = DprDataModule(
        pretrained_model_name_or_path=PRETRAINED_MODEL_NAME,
        batch_size=8,
    )
    data_module.setup("validate")
    batch = next(iter(data_module.val_dataloader()))

    assert len(batch) == 13
    assert batch["query_input_ids"].shape == (8, 512)
    assert batch["query_attention_mask"].shape == (8, 512)
    assert batch["query_token_type_ids"].shape == (8, 512)
    assert len(batch["queries"]) == 8

    assert batch["context_input_ids"].shape == (8, 512)
    assert batch["context_attention_mask"].shape == (8, 512)
    assert batch["context_token_type_ids"].shape == (8, 512)
    assert len(batch["contexts"]) == 8

    assert batch["bm25_input_ids"].shape == (8, 64, 512)
    assert batch["bm25_attention_mask"].shape == (8, 64, 512)
    assert batch["bm25_token_type_ids"].shape == (8, 64, 512)
    assert batch["bm25_labels"].shape == (8, 64)
    assert len(batch["bm25_contexts"]) == 8
