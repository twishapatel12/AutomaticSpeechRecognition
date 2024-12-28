# Import necessary libraries
import numpy as np
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import evaluate
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import TrainingArguments, Trainer
import gradio as gr

# Load the PolyAI dataset
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train[:80]")

# Remove unnecessary columns
dataset = dataset.remove_columns(['path', 'english_transcription', 'intent_class'])

# Split the dataset into train and test
dataset = dataset.train_test_split(test_size=0.2, shuffle=False)

# Declare device variable
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Resample the dataset to 16 Khz as model is trained on 16khz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Load Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.to(device)

# Let's process the first example of the train dataset
inputs = processor(dataset['train'][3]["audio"]["array"], sampling_rate=16000, return_tensors="pt")

# Getting the predictions
with torch.no_grad():
    logits = model(**inputs).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(f"Transcription for first audio: {transcription}")

def prepare_dataset(batch):
    audio = batch["audio"]

    # Process the audio input
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_values[0]

    # Process the transcription (labels)
    batch["labels"] = processor(
        text=batch["transcription"].upper(), return_tensors="pt", padding=True
    ).input_ids[0]

    return batch

# Apply the function to prepare the dataset
encoded_dataset = dataset.map(prepare_dataset, num_proc=1)

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_values = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_values, padding=self.padding, return_tensors="pt")

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")

# Load WER metric
wer = evaluate.load('wer')

# Compute metrics function for evaluation
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer_score = wer.compute(predictions=pred_str, references=label_str)
    return {"wer": wer_score}

# Define training arguments
training_args = TrainingArguments(
    output_dir="wav2vec2_finetuned",
    gradient_checkpointing=True,
    per_device_train_batch_size=1,
    learning_rate=1e-5,
    warmup_steps=2,
    max_steps=100,
    fp16=False,
    optim='adafactor',
    group_by_length=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=1,
    eval_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Get a sample from the test dataset
i2 = processor(dataset['test'][6]["audio"]["array"], sampling_rate=16000, return_tensors="pt")
print(f"The input test audio is: {dataset['test'][6]['transcription']}")

# Prediction for test data
with torch.no_grad():
    logits = model(**i2.to(device)).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(f'The output prediction is: {transcription[0]}')

# Define the transcription function for Gradio interface
def transcribe(audio):
    # Process the audio input
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_values
    inputs = inputs.to(device)

    # Get the logits from the model
    with torch.no_grad():
        logits = model(inputs).logits

    # Decode the logits to get the predicted transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]

# Create the Gradio interface
iface = gr.Interface(fn=transcribe, inputs=gr.Microphone(type="numpy"), outputs="text")
iface.launch()
