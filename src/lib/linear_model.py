import torch.nn as nn
import torch

class MCQCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        labels = torch.tensor([f["labels"] for f in features])

        max_len = max(len(choice) for f in input_ids for choice in f)

        padded = []
        for f in input_ids:
            choices = []
            for choice in f:
                pad_len = max_len - len(choice)
                choices.append(choice + [self.pad_token_id] * pad_len)
            padded.append(choices)

        return {
            "input_ids": torch.tensor(padded),
            "labels": labels
        }

class MCQModel(nn.Module):
    def __init__(self, model, num_choices):
        super().__init__()
        self.model = model
        self.linear = nn.Linear(model.config.hidden_size, 1)
        self.num_choices = num_choices


    def forward(self, input_ids, labels=None):
        B, C, T = input_ids.shape

        input_ids = input_ids.view(B * C, T)

        outputs = self.model(input_ids=input_ids)
        hidden = outputs.last_hidden_state  # (B*C, T, H)

        rep = hidden[:, -1, :]  # (B*C, H)

        logits = self.linear(rep)  # (B*C, 1)
        logits = logits.view(B, C)  # (B, C)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}