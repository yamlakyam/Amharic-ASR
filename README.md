**Introduction:**
This repository contains code for training an Automatic Speech Recognition (ASR) system for the Amharic language using the Wav2Vec2 model. The ASR system is trained using end-to-end learning techniques and incorporates data augmentation methods to improve performance.

**Data Preparation:**
The dataset used for training and evaluation should be organized in a specific directory structure. The audio files should be stored in separate directories for training and validation sets. The path to these directories should be specified in the code.

**Training:**
To train the ASR system, run the `Amharic ASR.py` script. You can specify various training parameters such as batch size, learning rate, and augmentation techniques using command-line arguments.


**Evaluation:**
After training, the model will be evaluated using the validation set. Evaluation metrics such as Word Error Rate (WER) and Character Error Rate (CER) will be computed and logged. 

**Results:**
The trained model will be saved in the specified output directory along with logs and evaluation results.

**References:**
- Wav2Vec2 Paper: [https://arxiv.org/abs/2006.11477](https://arxiv.org/abs/2006.11477)

Please refer to the code comments and documentation for more details on each component and functionality. If you encounter any issues or have questions, feel free to raise an issue in the repository.
