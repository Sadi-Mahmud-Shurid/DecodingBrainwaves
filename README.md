# DecodingBrainwaves

## 🚀 How to Reproduce

Follow these steps to set up and run this project.

### 1. Clone the Repository
```bash
git clone https://github.com/Sadi-Mahmud-Shurid/DecodingBrainwaves.git
cd DecodingBrainwaves
```

### 2. Download Preprocessed Data

Download the preprocessed EEG dataset from [Google Drive - Preprocessed EEG Data](https://drive.google.com/drive/folders/1XqV6MMl28iYXkQBMEFHfEXllGmCbqpOu) and place it in the same directory.

### 3. Install Dependencies

Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

### 4. Train the Hybrid EEG Encoder

Run the following training script to train the encoder:
```bash
python train_eeg_classifier.py \
  --eeg_dataset data/block/eeg_55_95_std.pth \
  --splits_path data/block/block_splits_by_image_all.pth \
  --output ./hybrid_eeg_encoder \
  --image_dir data/images/ \
  --batch_size 8 \
  --num_epochs 100 \
  --learning_rate 5e-5
```

### 5. Fine-tune with LLM

Run the fine-tuning script to integrate LLMs with EEG representations (using DeepSeek-LLM-7B-Chat):
```bash
python finetune_llm.py \
  --eeg_dataset data/block/eeg_55_95_std.pth \
  --splits_path data/block/block_splits_by_image_all.pth \
  --eeg_encoder_path ./hybrid_eeg_encoder \
  --image_dir data/images/ \
  --output deepseek_chat_hybrid_eeg_model \
  --llm_backbone_name_or_path deepseek-ai/deepseek-llm-7b-chat \
  --load_in_8bit \
  --bf16 \
  --batch_size 2 \
  --gradient_accumulation_steps 32
```

### 6. Run Inference

Use the trained model to generate words from EEG signals:
```bash
python inference.py \
  --model_path ./deepseek_chat_hybrid_eeg_model \
  --eeg_dataset data/block/eeg_55_95_std.pth \
  --splits_path data/block/block_splits_by_image_all.pth \
  --image_dir data/images/ \
  --dest results_hybrid_deepseek_chat.csv
```

### 7. Evaluate Results

To run the evaluation, execute the `metrics based evaluation.ipynb` notebook. 

> **Note:** Make sure to update the paths and file names according to your directory structure and the CSV file generated during inference.

## Acknowledgments

This repository builds upon and extends the excellent work by Abhijit Mishra and collaborators in the Thought2Text project.

💻 Foundational Codebase: [github.com/abhijitmishra/Thought2Text](https://github.com/abhijitmishra/Thought2Text)
