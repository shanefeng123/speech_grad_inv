from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizerFast
from datasets import load_dataset
from tqdm import tqdm
from utils import *

DEVICE = "cuda:1"
EPOCH = 10
batch_size = 32
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-tiny.en")
whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").to(DEVICE)

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset.remove_columns(["english_transcription", "intent_class", "lang_id"])

train_dataset_audio = [sample["array"] for sample in dataset["train"]["audio"]]
train_dataset_transcription = dataset["train"]["transcription"]
test_dataset_audio = [sample["array"] for sample in dataset["test"]["audio"]]
test_dataset_transcription = dataset["test"]["transcription"]

train_inputs = feature_extractor(raw_speech=train_dataset_audio, sampling_rate=16000,
                                 return_tensors="pt", return_attention_mask=True)
train_text = tokenizer(train_dataset_transcription, return_tensors="pt", padding=True, truncation=True,
                       return_attention_mask=True)
train_inputs["decoder_input_ids"] = train_text["input_ids"].clone()
train_inputs["decoder_attention_mask"] = train_text["attention_mask"].clone()
train_inputs["labels"] = train_text["input_ids"].clone()
train_dataset = MyAudioDataset(train_inputs)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_inputs = feature_extractor(raw_speech=test_dataset_audio, sampling_rate=16000,
                                return_tensors="pt", return_attention_mask=True)
test_text = tokenizer(test_dataset_transcription, return_tensors="pt", padding=True, truncation=True,
                      return_attention_mask=True)
test_inputs["decoder_input_ids"] = test_text["input_ids"].clone()
test_inputs["decoder_attention_mask"] = test_text["attention_mask"].clone()
test_inputs["labels"] = test_text["input_ids"].clone()

test_dataset = MyAudioDataset(test_inputs)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#
optimizer = torch.optim.AdamW(whisper.parameters(), lr=1e-5)

for i in range(EPOCH):
    train_loop = tqdm(train_dataloader)
    epoch_train_loss = 0
    epoch_train_score = 0
    epoch_test_loss = 0
    epoch_test_score = 0
    epoch_train_predictions = []
    epoch_train_labels = []
    epoch_test_predictions = []
    epoch_test_labels = []
    for batch in train_loop:
        whisper.train()
        train_loss, train_predictions, train_score, train_labels = train_batch(whisper, batch, DEVICE, optimizer, tokenizer)
        epoch_train_loss += train_loss.item()
        epoch_train_score += train_score
        epoch_train_predictions += train_predictions
        epoch_train_labels += train_labels
        train_loop.set_description(f"Epoch {i}")
        train_loop.set_postfix(train_loss=train_loss.item(), train_wer=train_score)

    test_loop = tqdm(test_dataloader)
    for batch in test_loop:
        whisper.eval()
        test_loss, test_predictions, test_score, test_labels = test_batch(whisper, batch, DEVICE, tokenizer)
        epoch_test_loss += test_loss.item()
        epoch_test_score += test_score
        epoch_test_predictions += test_predictions
        epoch_test_labels += test_labels
        test_loop.set_description(f"Epoch {i}")
        test_loop.set_postfix(test_loss=test_loss.item(), test_wer=test_score)

    average_train_loss = epoch_train_loss / len(train_dataloader)
    average_train_score = epoch_train_score / len(train_dataloader)
    average_test_loss = epoch_test_loss / len(test_dataloader)
    average_test_score = epoch_test_score / len(test_dataloader)
    print(f"Epoch {i} | Train Loss: {average_train_loss} | Train WER: {average_train_score} | "
          f"Test Loss: {average_test_loss} | Test WER: {average_test_score}")

#
# generated_ids = model.generate(inputs=train_input_features)
# #
# transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
# wer = evaluate.load("wer")
# score = wer.compute(predictions=transcriptions, references=train_dataset_transcription[:100])
# for transcription in transcriptions:
#     print(transcription)
