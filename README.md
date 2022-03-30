# **SPCUP2022 Submission**

## **Team ID: 27612**

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
