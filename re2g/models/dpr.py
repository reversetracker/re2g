import lightning as pl
import numpy as np
import torch
from lightning.pytorch.utilities import grad_norm
from torch import nn
from transformers import ElectraModel


class QueryEncoder(nn.Module):
    def __init__(
        self, pretrained_model_name_or_path: str, num_trainable_layers: int = 2
    ):
        super(QueryEncoder, self).__init__()
        self.electra = ElectraModel.from_pretrained(pretrained_model_name_or_path)

        for param in self.electra.parameters():
            param.requires_grad = False

        for count, layer in enumerate(
            self.electra.encoder.layer[-num_trainable_layers:]
        ):
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        sequence_output = outputs.last_hidden_state
        cls_token_output = sequence_output[:, 0, :]
        return cls_token_output


class ContextEncoder(nn.Module):
    def __init__(
        self, pretrained_model_name_or_path: str, num_trainable_layers: int = 2
    ):
        super(ContextEncoder, self).__init__()
        self.electra = ElectraModel.from_pretrained(pretrained_model_name_or_path)

        for param in self.electra.parameters():
            param.requires_grad = False

        for count, layer in enumerate(
            self.electra.encoder.layer[-num_trainable_layers:]
        ):
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        sequence_output = outputs.last_hidden_state
        cls_token_output = sequence_output[:, 0, :]
        return cls_token_output


class DPR(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_query_trainable_layers: int = 2,
        num_context_trainable_layers: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        super(DPR, self).__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.query_encoder = QueryEncoder(
            pretrained_model_name_or_path,
            num_trainable_layers=num_query_trainable_layers,
        )
        self.context_encoder = ContextEncoder(
            pretrained_model_name_or_path,
            num_trainable_layers=num_context_trainable_layers,
        )
        self.criteria = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
    ):
        query_embeddings = self.query_encoder(query_input_ids, query_attention_mask)
        context_embeddings = self.context_encoder(
            context_input_ids, context_attention_mask
        )
        query_embeddings_t = query_embeddings.transpose(0, 1)
        similarity_scores = torch.matmul(context_embeddings, query_embeddings_t)
        return similarity_scores, query_embeddings, context_embeddings

    def _calc_scores(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        bm25_input_ids: torch.Tensor | None,
        bm25_attention_mask: torch.Tensor | None,
        include_bm25: bool = True,
    ):

        # Memorize original shape first
        bm25_input_shape = bm25_input_ids.shape
        bm25_attention_mask_shape = bm25_attention_mask.shape

        # Reshape to 2D for encoding
        bm25_input_ids = bm25_input_ids.reshape(-1, bm25_input_shape[-1])
        bm25_attention_mask = bm25_attention_mask.reshape(
            -1, bm25_attention_mask_shape[-1]
        )

        # query embeddings
        query_embeddings = self.query_encoder(query_input_ids, query_attention_mask)

        # context embeddings
        context_embeddings = self.context_encoder(
            context_input_ids, context_attention_mask
        )

        # calculate in-batch scores
        in_batch_scores = torch.matmul(query_embeddings, context_embeddings.t())

        if not include_bm25:
            return in_batch_scores

        # bm25 embeddings
        bm25_embeddings = self.context_encoder(bm25_input_ids, bm25_attention_mask)

        # calculate bm25 scores
        bm25_embeddings_r = bm25_embeddings.reshape(
            bm25_input_shape[0], bm25_input_shape[1], -1
        )
        bm25_embeddings_t = bm25_embeddings_r.transpose(-2, -1)
        query_embeddings_r = query_embeddings.unsqueeze(1)

        bm25_scores = torch.matmul(query_embeddings_r, bm25_embeddings_t)
        bm25_scores = bm25_scores.squeeze(dim=1)

        # merge scores
        merged_scores = torch.cat((in_batch_scores, bm25_scores), dim=1)

        # The shape will be from N x M matrix to N x (M + K) matrix
        # k is the number of bm25 documents
        return merged_scores

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        query_input_ids = batch["query_input_ids"]
        query_attention_mask = batch["query_attention_mask"]
        context_input_ids = batch["context_input_ids"]
        context_attention_mask = batch["context_attention_mask"]
        bm25_input_ids = batch["bm25_input_ids"]
        bm25_attention_mask = batch["bm25_attention_mask"]
        batch_size = query_input_ids.shape[0]

        scores = self._calc_scores(
            query_input_ids,
            query_attention_mask,
            context_input_ids,
            context_attention_mask,
            bm25_input_ids,
            bm25_attention_mask,
            include_bm25=True,
        )
        loss = self._calc_loss(scores)
        self.log(
            name="train_loss",
            value=loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        query_input_ids = batch["query_input_ids"]
        query_attention_mask = batch["query_attention_mask"]
        context_input_ids = batch["context_input_ids"]
        context_attention_mask = batch["context_attention_mask"]
        bm25_input_ids = batch["bm25_input_ids"]
        bm25_attention_mask = batch["bm25_attention_mask"]
        batch_size = query_input_ids.shape[0]

        scores = self._calc_scores(
            query_input_ids,
            query_attention_mask,
            context_input_ids,
            context_attention_mask,
            bm25_input_ids,
            bm25_attention_mask,
            include_bm25=True,
        )

        loss = self._calc_loss(scores)
        mrr = self._calc_mrr(scores)
        self.log(
            name="val_loss",
            value=loss,
            logger=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            name="val_mrr",
            value=mrr,
            logger=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return {"val_loss": loss, "mrr": mrr}

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        """Calculate Mean Reciprocal Rank (MRR)"""
        pass

    # def on_before_optimizer_step(self, optimizer):
    #     norms = grad_norm(self.layer, norm_type=2)
    #     self.log_dict(norms)
    # trainer = Trainer(detect_anomaly=True)

    def _calc_loss(self, scores: torch.Tensor) -> torch.Tensor:
        labels = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        loss = self.criteria(scores, labels)
        return loss

    @staticmethod
    def _calc_mrr(scores: torch.Tensor) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""

        scores_mat = scores.cpu().detach().numpy()

        acc = 0.0
        for i in range(scores_mat.shape[0]):
            scores = scores_mat[i]
            scores = np.argsort(-scores)
            for j, rank in enumerate(scores):
                if rank == i:
                    acc += 1 / (j + 1)
                    break
        m = acc / scores.shape[0]
        return m
