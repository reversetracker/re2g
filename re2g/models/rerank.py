import lightning as L
import torch
import torch.nn as nn
from transformers import ElectraModel

from re2g.losses import MPNLWithLogitsLoss


class ReRanker(nn.Module):

    def __init__(
        self, pretrained_model_name_or_path: str, num_trainable_layers: int = 2
    ):
        super(ReRanker, self).__init__()

        self.electra = ElectraModel.from_pretrained(pretrained_model_name_or_path)
        self.linear_1 = nn.Linear(768, 256)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(256, 16)
        self.tanh = nn.Tanh()
        self.linear_3 = nn.Linear(16, 1)

        for param in self.electra.parameters():
            param.requires_grad = False

        for count, layer in enumerate(
            self.electra.encoder.layer[-num_trainable_layers:]
        ):
            for param in layer.parameters():
                param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=False,
        )
        sequence_output = outputs.last_hidden_state
        x = sequence_output[:, 0, :]
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.tanh(x)
        x = self.linear_3(x)
        return x


class Rerank(L.LightningModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_trainable_layers: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        loss_type: str = "mpnl",
    ):
        super(Rerank, self).__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.num_trainable_layers = num_trainable_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.reranker = ReRanker(pretrained_model_name_or_path, num_trainable_layers)

        if loss_type not in ["mpnl", "bce"]:
            raise ValueError("loss_type must be one of ['mpnl', 'bce']")

        if loss_type == "mpnl":
            self.loss = MPNLWithLogitsLoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

        self.save_hyperparameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ):
        logits = self.reranker(input_ids, attention_mask, token_type_ids)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]
        shape = input_ids.shape

        input_ids = input_ids.view(-1, shape[-1])
        attention_mask = attention_mask.view(-1, shape[-1])
        token_type_ids = token_type_ids.view(-1, shape[-1])

        logits = self.forward(input_ids, attention_mask, token_type_ids)
        logits = logits.view(labels.shape)

        train_loss = self.loss(logits, labels.float())
        self.log(
            name="train_loss",
            value=train_loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=shape[0],
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]
        shape = input_ids.shape

        input_ids = input_ids.view(-1, shape[-1])
        attention_mask = attention_mask.view(-1, shape[-1])
        token_type_ids = token_type_ids.view(-1, shape[-1])

        scores = self.forward(input_ids, attention_mask, token_type_ids)
        scores = scores.view(labels.shape)

        val_loss = self.loss(scores, labels.float())
        self.log(
            name="val_loss",
            value=val_loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=shape[0],
        )
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
