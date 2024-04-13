from datasets import load_dataset
import matplotlib.pyplot as plt

dataset = load_dataset("squad_kor_v1")

context_lengths = []

for item in dataset["train"]:
    context_length = len(item["context"])
    context_lengths.append(context_length)

plt.figure(figsize=(5, 6))
plt.hist(context_lengths, bins=200, color="blue", alpha=0.7)
plt.title("Context Lengths in KorQuAD Dataset")
plt.xlabel("Length of context")
plt.ylabel("Frequency")
plt.xlim(0, 2000)
plt.show()
