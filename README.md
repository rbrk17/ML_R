# ML_R

# Backdoor Detector for BadNets using Pruning Defense

Author: RAKSHANA B S (RB5118)

Please note that I have replaced variables as per homework instructions for clarity:
- 'B' refers to 'original_badnet'.
- 'G' refers to 'RepairedNet'.

## Introduction
This project focuses on designing a backdoor detector for BadNets trained on the YouTube Face dataset using the pruning defense technique discussed in class. The primary goal of the detector is to correctly classify inputs as either clean or backdoored. The input for the detector includes:

- `original_badnet (B)`: A backdoored neural network classifier with N classes.
- `Dvalid`: A validation dataset of clean, labeled images.

The objective is to output the correct class if the test input is clean (in the range [1, N]) or class N+1 if the input is backdoored.

##Data
## Data
You can access the dataset used in this project by clicking [here](https://drive.google.com/drive/folders/1MB_3IYnaWJF9E0V_MnIcq5IC6Q1lZl-s?usp=sharing).

## Running the Code

Follow these steps to run the project successfully:

1. **Open Python Environment**: Use either Jupyter Notebook or Google Colab as your Python environment.

2. **Upload the Notebook**: Ensure you have the notebook file named `Backdoor_rb5118.ipynb` available.

3. **Prepare Data Files**:
   - You'll need the following data files:
     - Clean validation data: `valid.h5`
     - Clean test data: `test.h5`
     - Poisoned (backdoored) test data: `bd_test.h5`
   - Make sure these data files are accessible and ready for use.

4. **Mount Google Drive (For Google Colab)**:
   - If you are using Google Colab, run the following code in a code cell to mount your Google Drive. This will grant access to your data files:

     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```

5. **Update File Paths**:
   - Locate the section in the notebook where file paths are defined.
   - Replace the placeholders with the actual paths to your data files in Google Drive:

     ```python
     # Path to the clean validation data
     valid_clean_data_path = '/content/drive/MyDrive/your-path-in-drive/valid.h5'
     # Path to the clean test data
     test_clean_data_path = '/content/drive/MyDrive/your-path-in-drive/test.h5'
     # Path to the poisoned (backdoored) test data
     test_poisoned_data_path = '/content/drive/MyDrive/your-path-in-drive/bd_test.h5'
     # Path to the model file (architecture)
     network_model_path = '/content/drive/MyDrive/your-path-in-drive/bd_net.h5'
     # Path to the model weights
     network_weights_path = '/content/drive/MyDrive/your-path-in-drive/bd_weights.h5'
     ```

   - Ensure that you provide the correct paths to your specific files.

By following these steps, you'll be able to execute the code using your own data files and model files. This flexible approach allows you to set up the project according to your environment and file locations.



## Method


### Pruning Defense
We implemented the pruning defense by selectively removing channels in the last pooling layer of `original_badnet (B)`, just before the fully connected layers. Channels are removed one at a time, starting with those having the highest average activation values over the entire validation set. Pruning continues until the validation accuracy drops by a predefined percentage. The pruned network becomes the new network 'B'.

### Goodnet G
Our 'Goodnet G' works as follows for each test input:

- Run the input through both 'original_badnet (B)' and the pruned network 'B'.
- If the classification outputs of 'B' and 'B'' are the same (i.e., class 'i'), the detector outputs class 'i'.
- If the outputs differ, the detector outputs class 'N+1'.

## Results
Below is the evaluation table for the repaired networks:

| Model         | Repaired Clean Accuracy | Attack Success Rate |
|---------------|-------------------------|---------------------|
| 2% repaired   | 95.7443                 | 100                 |
| 4% repaired   | 92.1278                 | 99.9844             |
| 10% repaired  | 84.3336                 | 77.2097             |


