import copy
import time

import torch.nn
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizerFast
from datasets import load_dataset
from utils import *
from argparse import ArgumentParser
import random
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="openai/whisper-tiny.en")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--dataset", type=str, default="PolyAI/minds14")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--recover_batch", default=0, type=int, help="batch to recover from")
parser.add_argument("--run", default="first", type=str, help="number of run")
args = parser.parse_args()

random.seed(42)
torch.manual_seed(42)

DEVICE = args.device if (torch.cuda.is_available() or torch.has_mps) else "cpu"
BATCH_SIZE = args.batch_size
RECOVER_BATCH = args.recover_batch

server = WhisperForConditionalGeneration.from_pretrained(args.model).to(DEVICE)
client = copy.deepcopy(server)
feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)
tokenizer = WhisperTokenizerFast.from_pretrained(args.model)

# Prepare the data
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
dataset = dataset.remove_columns(["english_transcription", "intent_class", "lang_id"])
train_dataset_audio = [sample["array"] for sample in dataset["audio"]]
train_dataset_transcription = dataset["transcription"]
train_inputs = feature_extractor(raw_speech=train_dataset_audio, sampling_rate=16000,
                                 return_tensors="pt", return_attention_mask=True, do_normalize=True)

train_text = tokenizer(train_dataset_transcription, return_tensors="pt", padding=True, truncation=True,
                       return_attention_mask=True)
train_inputs["decoder_input_ids"] = train_text["input_ids"].clone()
train_inputs["decoder_attention_mask"] = train_text["attention_mask"].clone()
train_inputs["labels"] = train_text["input_ids"].clone()
train_dataset = MyAudioDataset(train_inputs)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loop = tqdm(train_dataloader, leave=True)

optimizer = torch.optim.AdamW(server.parameters(), lr=1e-5)
batch_num = 1

for batch in train_loop:
    server = copy.deepcopy(client)
    client.train()
    train_loss, train_predictions, train_score, train_labels = train_batch(client, batch, DEVICE, optimizer, tokenizer)
    if batch_num <= RECOVER_BATCH:
        batch_num += 1
        continue

    client_grads = []
    for param in client.parameters():
        if param.grad is not None:
            client_grads.append(param.grad.clone())

    # input_features = batch["input_features"].clone().detach().to(DEVICE)
    labels = batch["labels"].clone().detach().to(DEVICE)
    attention_mask = batch["attention_mask"].clone().detach().to(DEVICE)
    decoder_attention_mask = batch["decoder_attention_mask"].clone().detach().to(DEVICE)

    spectrogram_lengths = []
    for i in range(BATCH_SIZE):
        spectrogram_lengths.append(torch.sum(attention_mask[i]).item())

    transcript_lengths = []
    for i in range(BATCH_SIZE):
        transcript_lengths.append(torch.sum(decoder_attention_mask[i]).item())

    server.eval()
    continuous_optimizer = ContinuousOptimizer(client_grads, BATCH_SIZE, server, client, DEVICE, tokenizer, lr=0.01,
                                               num_of_iter=2000, labels=labels, attention_mask=attention_mask,
                                               decoder_attention_mask=decoder_attention_mask,
                                               spectrogram_lengths=spectrogram_lengths,
                                               transcript_lengths=transcript_lengths, alpha=0.01)
    recovered_mel_spectrogram = continuous_optimizer.optimize()

    recovered_ids = server.generate(input_features=recovered_mel_spectrogram)
    generate_ids = server.generate(input_features=batch["input_features"].to(DEVICE))

    recovered_transcript = tokenizer.batch_decode(recovered_ids, skip_special_tokens=True)
    original_transcript = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

    for i in range(BATCH_SIZE):
        print("Original transcript: ", original_transcript[i])
        print("Recovered transcript: ", recovered_transcript[i])

    MSE = torch.nn.MSELoss()
    mse = MSE(recovered_mel_spectrogram, batch["input_features"].to(DEVICE))
    rmse = torch.sqrt(mse)
    print("RMSE:", rmse)
    break
