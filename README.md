# Speech Recognition Project

## Overview
This project implements a speech recognition system using Python and the Wav2Vec2 model from Hugging Face. The system is fine-tuned on the PolyAI MINDS-14 dataset and deployed with a real-time transcription interface using Gradio.

## Features
- **Dataset**: Utilizes the PolyAI MINDS-14 dataset, resampled to 16kHz.
- **Model & Processor**: Employs the Wav2Vec2ForCTC model and processor for audio-to-text transcription.
- **Training**: Fine-tunes the model with custom training arguments and a data collator for handling varying input lengths.
- **Evaluation**: Assesses model performance using the Word Error Rate (WER) metric.
- **Deployment**: Provides a Gradio interface for real-time transcription.

## Skills Used
- Python
- Machine Learning
- Deep Learning
- Natural Language Processing (NLP)
- Audio Processing
- PyTorch
- Transformers (Hugging Face)
- Gradio Interface Development
- Model Evaluation (WER)
- Dataset Management
- GPU Acceleration (CUDA)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/speech-recognition-project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd speech-recognition-project
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Training the Model
1. Run the training script to fine-tune the model:
   ```bash
   python train.py
   ```

### Real-Time Transcription
1. Launch the Gradio interface:
   ```bash
   python app.py
   ```
2. Use the microphone input to test real-time transcription.

## Example
Transcription for a sample audio:
```python
The input test audio is: "Sample transcription"
The output prediction is: "Sample prediction"
```

## Acknowledgements
- Hugging Face for providing the Wav2Vec2 model and transformers library.
- PolyAI for the MINDS-14 dataset.
- Gradio for the interactive interface.


