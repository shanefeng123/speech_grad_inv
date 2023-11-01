import evaluate
import torch


class MyAudioDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return {key: torch.tensor(val[index]) for key, val in self.data.items()}

    def __len__(self):
        return len(self.data["input_features"])


def train_batch(model, batch, device, optimizer, tokenizer):
    optimizer.zero_grad()
    input_features = batch["input_features"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    decoder_input_ids = batch["decoder_input_ids"].to(device)
    decoder_attention_mask = batch["decoder_attention_mask"].to(device)
    train_outputs = model(input_features=input_features, attention_mask=attention_mask, labels=labels,
                          decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
    train_loss = train_outputs.loss
    train_loss.backward()
    optimizer.step()
    predictions = torch.argmax(train_outputs.logits, dim=-1)
    wer = evaluate.load("wer")
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    score = wer.compute(predictions=predictions, references=labels)
    return train_loss, predictions, score, labels


def test_batch(model, batch, device, tokenizer):
    input_features = batch["input_features"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    decoder_input_ids = batch["decoder_input_ids"].to(device)
    decoder_attention_mask = batch["decoder_attention_mask"].to(device)
    test_outputs = model(input_features=input_features, attention_mask=attention_mask, labels=labels,
                         decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
    test_loss = test_outputs.loss
    predictions = torch.argmax(test_outputs.logits, dim=-1)
    wer = evaluate.load("wer")
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    score = wer.compute(predictions=predictions, references=labels)
    return test_loss, predictions, score, labels