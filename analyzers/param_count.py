# BERT-base 모델 로드
from transformers import BertModel

# google
model = BertModel.from_pretrained("bert-base-uncased")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")
print(model)
