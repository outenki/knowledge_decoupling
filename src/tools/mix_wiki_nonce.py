from datasets import load_from_disk, concatenate_datasets, DatasetDict
from pathlib import Path

data_path = Path("/home/pj25000107/ku50001566/projects/knowledge_decoupling/input")
nonce_path = data_path / "wikimedia-nonce-bs1024"
wiki_path = data_path / "wikimedia-bs1024"


nonce = load_from_disk(str(nonce_path))
wiki = load_from_disk(str(wiki_path))["train"].train_test_split(test_size=0.1, seed=42)

train_size = len(nonce['train']) * 2
val_size = len(nonce['val']) * 2

sub_train = wiki['train'].shuffle(seed=42).select( range(train_size))
sub_val = wiki['test'].shuffle(seed=42).select(range(val_size))

merged_train = concatenate_datasets([nonce['train'], sub_train]).shuffle(seed=42)
merged_val = concatenate_datasets([nonce['val'], sub_val]).shuffle(seed=42)
merged = DatasetDict({
    'train': merged_train,
    'val': merged_val
})
out_path = data_path / "wikimedia-nonce-mix-bs1024-2m"
merged.save_to_disk(str(out_path))
print(f"Saved merged dataset to {out_path}")
