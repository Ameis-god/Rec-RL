# ExplanRec: Explainable Recommendation with Large Language Models

ExplanRec is a novel framework that combines SASRec (Self-Attentive Sequential Recommendation) with Large Language Models to provide not only accurate recommendations but also reasonable explanations for the recommendations. The system integrates traditional collaborative filtering with the reasoning capabilities of LLMs to enhance both recommendation performance and interpretability.

## 🚀 Key Features

- **Hybrid Architecture**: Seamlessly integrates SASRec sequential recommendation model with Large Language Models
- **Explainable Recommendations**: Generates natural language explanations for recommended items
- **Multi-task Framework**: Supports both recommendation accuracy and explanation generation tasks
- **End-to-End Pipeline**: Complete workflow from SASRec training to explanation generation
- **Flexible Model Support**: Compatible with various LLMs (e.g., Llama-3-8B-Instruct)

## 🏗️ System Architecture

The ExplanRec framework operates in two main stages:

### Stage 1: SASRec Foundation
1. **SASRec Training**: Train the SASRec model on sequential recommendation data
2. **SASRec Inference**: Generate item embeddings and recommendation results
3. **Data Preparation**: Process SASRec outputs for LLM training

### Stage 2: LLM Integration
1. **Data Pipeline**: Convert recommendation data into instruction-following format
2. **LLM Fine-tuning**: Train LLM with SASRec embeddings and recommendation tasks
3. **Model Alignment**: Perform alignment tasks for recommendation understanding
4. **Explanation Generation**: Generate natural language explanations for recommendations

## 🛠️ Environment Setup

### Prerequisites
- Python 3.10.14
- CUDA-compatible GPU
- Conda package manager

### Installation

```bash
# Create and activate conda environment
conda create -n ExplanRec python==3.10.14
conda activate ExplanRec

# Install PyTorch with CUDA support
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Data Preparation
1. Prepare your recommendation dataset in the required format
2. Place data files in the `data/` directory
3. Update path configurations in shell scripts

### Basic Usage
Run the complete pipeline using the provided shell scripts in sequential order:

```bash
# Navigate to project directory
cd ExplanRec

# Step 1-7: Execute shell scripts in order (see Detailed Usage section)
```

## 📋 Detailed Usage

The ExplanRec framework uses a series of shell scripts for different stages. **Important**: Configure all paths in the shell scripts before execution.

### Step 1: SASRec Model Training
```bash
bash shell/unirec_sasrec_train.sh
```
- Trains the SASRec model on sequential recommendation data
- Outputs trained model checkpoints
- Configure dataset name, learning rate, and other hyperparameters in the script

### Step 2: SASRec Model Inference
```bash
bash shell/unirec_sasrec_infer.sh
```
- Generates recommendations using the trained SASRec model
- Produces item embeddings and similarity files
- Creates training and testing recommendation data

### Step 3: Data Preprocessing Pipeline
```bash
bash shell/recexplainer_data_pipeline.sh
```
- Processes SASRec outputs for LLM training
- Generates instruction-following datasets
- Creates alignment and explanation training data
- Integrates with external APIs (e.g., Gemini) if needed

### Step 4: LLM Training
```bash
bash shell/train.sh
```
- Fine-tunes the base LLM with recommendation tasks
- Integrates SASRec embeddings into LLM training
- Supports multi-task learning (recommendation + explanation)
- Uses LoRA/PEFT for efficient fine-tuning

### Step 5: Model Merging
```bash
bash shell/merge.sh
```
- Merges the fine-tuned LoRA weights with the base model
- Prepares the final model for inference
- Integrates recommendation model components

### Step 6: Alignment Inference
```bash
bash shell/infer_alignment.sh
```
- Performs various alignment tasks:
  - Next item prediction (`uid2next`)
  - Item recovery (`uid2hist`)
  - Item ranking (`uidiid2rank`)
  - Interest classification (`uidiid2binary`)

### Step 7: Explanation Generation
```bash
bash shell/infer_explan.sh
```
- Generates natural language explanations for recommendations
- Produces interpretable recommendation results
- Supports case study analysis

## 📁 Project Structure

```
ExplanRec/
├── src/                    # Source code
│   ├── sft_training.py     # LLM fine-tuning script
│   ├── inference.py        # Inference engine
│   └── merge.py           # Model merging utilities
├── sasrec/                # SASRec model implementation
│   ├── unirec/            # UniRec framework
│   └── data/              # SASRec datasets
├── shell/                 # Shell scripts for pipeline execution
│   ├── *.sh              # Individual pipeline scripts
│   └── config/           # Configuration files
├── preprocess/           # Data preprocessing scripts
│   ├── amazon_generate_v3.py    # Data generation
│   ├── explan_data_gen.py       # Explanation data preparation
│   └── gemini_api.py           # External API integration
├── evaluate/             # Evaluation scripts and metrics
├── data/                 # Main data directory
│   └── Books/           # Example dataset
└── requirements.txt      # Python dependencies
```

## 🔧 Configuration

### Path Configuration
Before running any script, update the following paths:

```bash
# In each shell script, modify:
HOME=/your/home/path                    # Your home directory
model_name_or_path=/your/base_model/path    # Base LLM path
rec_model_name_or_path=/your/rec_model/path # SASRec model path
output_dir=/your/output/path             # Output directory
```

### Key Parameters

#### SASRec Training
- `DATASET_NAME`: Dataset identifier (e.g., "books")
- `learning_rate`: Learning rate for SASRec training
- `embedding_size`: Embedding dimension (default: 256)
- `max_seq_len`: Maximum sequence length (default: 200)

#### LLM Training
- `model_max_length`: Maximum input length for LLM (default: 1024)
- `max_hist_len`: Maximum history length (default: 9)
- `task_type`: Training task type ("both" for multi-task)
- `template_name`: Prompt template (e.g., "llama-3")

## 📊 Data Format

### Input Data Requirements
- **Sequential Data**: User interaction sequences in CSV format
- **Metadata**: Item metadata in JSON format
- **User History**: User-item interaction history
- **Item Similarities**: Pre-computed item similarity matrices

### SASRec Data Format
The SASRec model expects data in the following format:
- `user_history.csv`: User interaction sequences
- `train_ids.csv` / `test_ids.csv`: Train/test user splits
- Item metadata files for embedding initialization

## ⚠️ Important Notes

1. **Pre-trained SASRec**: The SASRec model data and checkpoints are directly provided - no need for separate data collection
2. **Path Configuration**: All scripts require careful path configuration before execution
3. **GPU Requirements**: The framework requires CUDA-compatible GPUs for training and inference
4. **Sequential Execution**: Shell scripts must be executed in the specified order
5. **Memory Requirements**: LLM training requires significant GPU memory (recommended: 24GB+)

## 🤝 Contributing

We welcome contributions to ExplanRec! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Submit a pull request with detailed description

### Reporting Issues
- Use GitHub Issues for bug reports
- Provide detailed environment information
- Include error logs and reproduction steps

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🔗 Related Work

This project builds upon:
- **SASRec**: Self-Attentive Sequential Recommendation
- **UniRec**: Universal Recommendation Framework  
- **LLaMA**: Large Language Model Meta AI

For questions and support, please open an issue on GitHub.