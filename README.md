# ImageCaptionGeneratorLSTM
Image caption generator baseline model using Long Short Term Memory

1. data_preprocessing.py - This file is responsible for preprocessing the Flickr8K dataset and generating the required input data files for training and testing the model. To execute this file, run the following command in your terminal:

python data_preprocessing.py --dataset_path /path/to/flickr8k_dataset --output_path /path/to/output_dir

Make sure to replace /path/to/flickr8k_dataset with the path to the Flickr8K dataset on your local machine and /path/to/output_dir with the path to the directory where you want to save the preprocessed data files.

2. encoder.py - This file contains the implementation of the encoder network that is responsible for generating feature vectors from input images. This file does not need to be executed separately, as it is imported by other files that require the encoder network.

3. decoder.py - This file contains the implementation of the decoder network that is responsible for generating captions from the feature vectors generated by the encoder network. This file does not need to be executed separately, as it is imported by other files that require the decoder network.

4. train.py - This file is responsible for training the image captioning model on the preprocessed data. To execute this file, run the following command in your terminal:

python train.py --data_path /path/to/preprocessed_data --output_path /path/to/output_dir --epochs 10 --batch_size 64

Make sure to replace /path/to/preprocessed_data with the path to the directory where the preprocessed data files are saved, /path/to/output_dir with the path to the directory where you want to save the trained model weights and other outputs, 10 with the number of epochs you want to train the model for, and 64 with the batch size you want to use during training.

5. test.py - This file is responsible for generating captions for test images using the trained model. To execute this file, run the following command in your terminal:

python test.py --data_path /path/to/preprocessed_data --model_path /path/to/trained_model --output_path /path/to/output_dir
Make sure to replace /path/to/preprocessed_data with the path to the directory where the preprocessed data files are saved, /path/to/trained_model with the path to the trained model weights file, and /path/to/output_dir with the path to the directory where you want to save the generated captions.

6. evaluation.py - This file is responsible for evaluating the performance of the trained model using BLEU score. To execute this file, run the following command in your terminal:

python evaluation.py --data_path /path/to/preprocessed_data --model_path /path/to/trained_model
Make sure to replace /path/to/preprocessed_data with the path to the directory where the preprocessed data files are saved and /path/to/trained_model with the path to the trained model weights file.

Note: It is important to ensure that the paths specified in the commands are correct and point to the correct directories/files on your local machine.





