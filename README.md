# Readme

This repository outlines the steps to set up the environment, prepare the SFT dataset, and run Supervised Fine-Tuning (SFT) and GRPO training.

## Phase 1: Environment Setup

1.Create a Conda Virtual Environment:

It is recommended to create a new conda environment to manage dependencies. Replace `<your_env_name>` with your preferred environment name.

```bash
conda create -n <your_env_name> python=3.12.9  
conda activate <your_env_name>  
```

2.Install Dependencies:

 Install the required packages using the `requirements.txt` file. Ensure you have PyTorch version 2.6.0 or compatible, as specified in the requirements.

```
pip install -r requirements.txt  
```

## Phase 2: SFT Dataset Construction

**Note**: You can skip this phase by directly using the pre-built datasets available in the `sft_datasets` directory. This phase may be time-consuming.


This phase involves generating and processing the dataset for Supervised Fine-Tuning.

1.Navigate to the  Directory:

```
cd sft_dataset_construction
```

2.Generate Initial Dataset: Execute the `generate.sh` script. This script utilizes a pre-trained explanation model to generate explanations, resulting in the `generate_dataset.json` file.

```
bash scripts/generate.sh  
```

3.Summarize and Refine Dataset: Run the `summary.sh` script. This script processes `generate_dataset.json` to produce a summarized and refined dataset, `summary_dataset.json`.

```
bash scripts/summary.sh  
```

4.Format Dataset: Execute the `extract_dataset.py` script to perform necessary formatting on the dataset.

```
python scripts/extract_dataset.py  
```

5.Split Dataset: Finally, run `divide_dataset.py` to split the processed dataset into training and validation sets.

```
python scripts/divide_dataset.py  
```

## Phase 3: Supervised Fine-Tuning (SFT)

To perform Supervised Fine-Tuning on the prepared dataset:

Execute SFT Script:

```
cd ..
bash sft.sh  
```

## Phase 4: GRPO Training

For GRPO training:

Execute Training Script:

```
bash train.sh  
```

## Phase 5: Inference

To run inference with the trained model:

Execute Inference Script:

```
bash inference.sh  
```