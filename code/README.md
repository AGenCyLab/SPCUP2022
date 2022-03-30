# **SPCUP2022 Submission**

## **Team ID: 27612**

## Contents of the folders
- `code` - The implementation of the synthetic speech algorithm detector. 
- `report` - The technical report.
- `scores` - The predictions of the model on the evaluation datasets parts 1 and 2 in `csv` format.

## Development Environment
1. It is recommended to use [Anaconda](https://www.anaconda.com/) to create a virtual environment before proceeding to run code.

```
conda create -n spcup_27612 python=3.8
conda activate spcup_27612
```

2. Python 3.8.12 is the version used for implementation.
3. After activating the newly created environment, the requirements can be installed using

```
cd code && pip install -r requirements.txt
```

The following sections assume that 
- The CUDA binaries are properly installed on the system.
- If using Anaconda, the correct versions of `cudatoolkit` and `cudnn` are installed in the corresponding environment according to the system.
- In the terminal, the user has changed the directory to `code/` and activated the proper environment before running any commands below.

## Performing inference on directory of `.wav` files

`code/infer.py` should be used to perform inference on a given directory containing `.wav` files for generating predictions `.csv` file that contains mapping between filename and predicted label. It supports the following command line arguments:

```
--dataset-path DATASET_PATH (required)
  The path to the directory containing the wav files to be used for inference
--training-config-file-path TRAINING_CONFIG_FILE_PATH (optional)
--model-checkpoint-path MODEL_CHECKPOINT_PATH (optional)
  The path to the checkpoint to use for inference
--num-workers NUM_WORKERS (optional)
  Number of processes to use for data loading
--answer-path ANSWER_PATH (optional)
  The path in which the answer csv file will be placed
```

By default, the script will generate the `score.csv` file and put it inside a subfolder under `scores` named with the current timestamp on the system.

```
python infer.py --dataset-path <path-to-wav-file-directory>
```

The rest of the options can be omitted as they have default values defined, but can be overriden if needed.

## Training the detector

For training, `train_tssdnet.py` is to be used. It accepts the following command line arguments:

```
--dataset-config-file-path DATASET_CONFIG_FILE_PATH (Optional)
  Path to dataset.yaml file
--training-config-file-path TRAINING_CONFIG_FILE_PATH (Optional)
  Path to train_params.yaml file
--checkpoint-path CHECKPOINT_PATH (Optional)
  Path to the folder where the checkpoints should be stored.
--gpu-indices GPU_INDICES (Optional)
  A comma separated list of GPU indices. Set as value of CUDA_VISIBLE_DEVICES environment variable. Defaults to "0".
--epochs EPOCHS (Required)
  Number of epochs to train the model for
--resume-from-checkpoint RESUME_FROM_CHECKPOINT (Optional)
  The path to the checkpoint from which to resume training from, if any.
--include-augmented-data (Optional)
  Boolean flag to indicate whether augmented data should be used for training.
--include-unseen-in-training-data (Optional)
  Boolean flag to indicate whether the class labelled '5' should be used for training.
```

The required datasets are automatically downloaded, given the existence of a properly formatted `config/dataset.yaml` file. The dataset is then divided into stratified, non-overlapping sets of training, validation and testing sets.

```
python train_tssdnet.py --checkpoint-path ./checkpoints/inc/ --gpu-indices 0,1 --epochs 200 --include-unseen-in-training-data --include-augmented-data
```

## Testing the detector
For testing, `test_tssdnet.py` is to be used. It accepts the following command line arguments:

```
--dataset-config-file-path DATASET_CONFIG_FILE_PATH (Optional)
  Path to dataset.yaml file
--training-config-file-path TRAINING_CONFIG_FILE_PATH (Optional)
  Path to train_params.yaml file
--model-checkpoint-path MODEL_CHECKPOINT_PATH (Required)
  The path to the model checkpoint to use for inference
--load-augmented-data (Optional)
  Load part 2 of the evaluation dataset
--submission-path SUBMISSION_PATH (Optional)
  The path in which the submission text file will be placed
```

```
python test_tssdnet.py --model-checkpoint-path ./checkpoints/tssdnet/inc_tssdnet_with_unseen_aug/last.ckpt --load-augmented-data
```

## Notebooks
The notebooks contain code to generate confusion matrices, ROC curves and Precision-Recall curves for the trained detector on a held-out subset of the training data.


## Accuracies

| Methods                       | Features                                    | Accuracy on  unaugmented  test set | Accuracy on  augmented  test set |
|-------------------------------|---------------------------------------------|------------------------------------|----------------------------------|
| ResNet34                      | Mel-spectrogram                             |                0.95                |               0.97               |
| ResNet18                      | Mel-spectrogram                             |              **0.98**              |               0.97               |
| VGG16                         | Mel-spectrogram                             |                0.70                |               0.68               |
| Inc-TSSDNet                   | Raw Waveform                                |                0.96                |             **1.00**             |
| Res-TSSDNet                   | Raw Waveform                                |                0.93                |               0.99               |
| Support Vector  Machine (SVM) | Mel-frequency  cepstral coefficients (MFCC) |                0.86                |               0.97               |
| Gaussian Mixture Model (GMM)  | Mel-frequency  cepstral coefficients (MFCC) |                0.25                |               0.39               |

## Contributions
 - Mahieyin Rahmun - mahieyin.rahmun@gmail.com
 - Zulker Nayeen Nahiyan - 1910063@iub.edu.bd
 - Sadia Khan - 1831231@iub.edu.bd
 - Rafat Hasan Khan - rafathasankhan@gmail.com
 - Rakibul Hasan Rajib - 1921834@iub.edu.bd
 - Mir Sayad Bin Almas - 1821778@iub.edu.bd
 - Dr. Sakira Hassan - sakira.hassan@gmail.com
 - Tanjim Taharat Aurpa - taurpa22@gmail.com
