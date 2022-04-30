# Reproducibility Project for DeepNote-GNN

April 25th 2022

*This is the code repository for the final project of UIUC's CS598 Deep Learning for Healthcare.*

This is the reproduced code for the paper *DeepNote-GNN: predicting hospital readmission using clinical notes and patient network* ([link](https://dl.acm.org/doi/10.1145/3459930.3469547)) by Sara Nouri Golmaei and Xiao Luo. The proposed DeepNote model leverages the large-scale pretrained ClinicalBERT ([link](https://arxiv.org/abs/1904.05342)) and Graph Convolutional Network (GCN) on the constructed patient network to jointly model readmission rate. 

This repository implemented the DeepNote model as well as all the baseline models (finetuning ClinicalBERT, bag-of-word, BiLSTM with word2vec) mentioned in the paper. We implement the models with PyTorch, [PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) for graph neural networks, and HuggingFace's [Transformers](https://huggingface.co/docs/transformers/index) for the pretrained ClinicalBERT.



## Main results

The performance on the `discharge` subset is shown in the below table.

| Model        | AUROC      | AURPC      | R@P80%     |
| ------------ | ---------- | ---------- | ---------- |
| DeepNote     | **0.8596** | **0.8520** | 0.6759     |
| DeepNote-GAT | 0.8591     | 0.8403     | **0.6793** |
| ClinicalBert | 0.7721     | 0.7730     | 0.2899     |
| BiLSTM       | 0.5318     | 0.5778     | 0.0942     |
| Bag-of-Word  | 0.6899     | 0.7147     | 0.1744     |

The performance on the `3day` subset is shown in the below table.

| Model        | AUROC      | AURPC      | R@P80%     |
| ------------ | ---------- | ---------- | ---------- |
| DeepNote     | 0.6345     | 0.6140     | 0.0380     |
| DeepNote-GAT | 0.6338     | 0.6211     | 0.0342     |
| ClinicalBert | 0.6066     | 0.6217     | **0.2421** |
| BiLSTM       | 0.5054     | 0.5359     | 0.1159     |
| Bag-of-Word  | **0.6503** | **0.6688** | 0.1377     |



## Requirements

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```



## Preparing the data

The paper used the [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) dataset, a large, freely-available database comprising de-identified health-related data. The dataset cannot be redistributed, so you should apply for the access to the dataset. Only the `ADMISSIONS.csv` and `NOTEEVENTS.csv` are needed, which are 1.1GB total in size.

The data are then preprocessed to extract the readmission information. We adopted the script from ClinicalBERT's [repo](https://github.com/kexinhuang12345/clinicalBERT), as was done by the authors of the paper. Run the preprocessing script with the following command:

```bash
python preprocess.py
```

The script will generate two subset, called `discharge` and `3day` respectively under the `data_root` folder. The usage is shown below.

```bash
usage: preprocessing script
optional arguments:
  -h, --help	show this help message and exit
  --data-root	DATA_ROOT (default: './data/')
  --seed		SEED (default: 1)
```

Then, use the pretrained ClinicalBERT to extract the representation for each clinical note. The pretrained weights are available at HuggingFace's model hub ([link](https://huggingface.co/AndyJ/clinicalBERT)). However, the config file is missing the `model_type` key, which will cause an error if directly loaded from the model hub. So you need to download the model and add `"model_type": "bert"` to `config.json`. After that, run the pretrained model using:

```bash
python run_pretrained.py
```

The script will generate `train.pt`, `val.pt`, and `test.pt` under the data folder. The usage is shown below:

```bash
usage: Running pretrained clinicalBERT
optional arguments:
  -h, --help	show this help message and exit
  --data-root 	DATA_ROOT (default: './data/discharge/')
  --model-dir	MODEL_DIR (default: './checkpoints/clinicalbert/')
  --batch-size	BATCH_SIZE (default: 128)
  --max-length	MAX_LENGTH (default: 512)
```



## Training the model

Using `train.py` to train the DeepNote model as well as the baseline models. For example, you can run the following script to train the DeepNote model:

```bash
python train.py configs/deepnote.yml --savename deepnote
```

The usage is shown below:

```bash
positional arguments:
  config
optional arguments:
  -h, --help	show this help message and exit
  --device		model device (default: 'cuda')
  --logdir		log folder for tensorboard and saving the model (default: './logs')
  --savename	log name for tensorboard (default: 'test')
  --resume		resume checkpoint (default: None)
```

The `configs` folder also includes the config file for finetuning ClinicalBERT and training BiLSTM model and the GAT version of DeepNote. Most of the arguments in the config file is self-explanatory. You can modify them to try different hyperparameter settings.

To run BiLSTM with word2vec, you should first extract the word2vec weights with [gensim](https://radimrehurek.com/gensim/) by running the following script:

```bash
pip install gensim
python gen_word2vec.py
```

The usage is shown below:

```bash
usage: Generate word2vec embeddings
optional arguments:
  -h, --help	show this help message and exit
  --save-path	SAVE_PATH (default: './checkpoints/word2vec/')
```

To run the bag-of-word model, use the following script:

```bash
python run_bow.py
```

The usage is shown below:

```bash
usage: Running Bag-of-Word model with logistic regression
optional arguments:
  -h, --help	show this help message and exit
  --data-root	DATA_ROOT (default: './data/discharge/')
  --max-feat	MAX_FEAT (default: 5000)
```



## Trying different threshold for building graph

We also provide the code for testing the impact of different thresholds for building graph on the final performance. Run the following command:

```bash
python run_diff_threshold.py
```

The script will generate a CSV file `diff_thres.csv` with the following fields:

```bash
['threshold', 'num_edges', 'sparsity', 'test_loss', 'acc', 'roc', 'prc', 'rp80']
```

The usage is shown below:

```bash
usage: Test different thresholds for building graph
positional arguments:
  config
optional arguments:
  -h, --help	show this help message and exit
  --thres-begin	THRES_BEGIN (default: 0.)
  --thres-end	THRES_END (default: 0.999)
  --num-thres	NUM_THRES (default: 50)
  --device		DEVICE (default: 'cuda')
```

