import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_data(path, num_samples):
    # Load data from file
    lines = pd.read_csv(path, names=["src", "tar"], sep="\t", nrows=num_samples)

    # Split data into source and target sequences
    source_texts = lines["src"]
    target_texts = ["\t" + text + "\n" for text in lines["tar"]]

    # Tokenize source sequences
    src_tokenizer = Tokenizer()
    src_tokenizer.fit_on_texts(source_texts)
    num_encoder_tokens = len(src_tokenizer.word_index) + 1
    encoder_input_sequences = src_tokenizer.texts_to_sequences(source_texts)
    encoder_input_sequences = pad_sequences(encoder_input_sequences, padding="post")

    # Tokenize target sequences
    tar_tokenizer = Tokenizer()
    tar_tokenizer.fit_on_texts(target_texts)
    num_decoder_tokens = len(tar_tokenizer.word_index) + 1
    decoder_input_sequences = tar_tokenizer.texts_to_sequences(target_texts)
    decoder_input_sequences = pad_sequences(decoder_input_sequences, padding="post")

    # Create decoder output sequences (shifted by 1)
    decoder_output_sequences = np.zeros_like(decoder_input_sequences)
    decoder_output_sequences[:, 0:-1] = decoder_input_sequences[:, 1:]
    decoder_output_sequences[:, -1] = 0

    # Split data into train and validation sets
    encoder_input_train, encoder_input_val, decoder_input_train, decoder_input_val, decoder_output_train, decoder_output_val = train_test_split(
        encoder_input_sequences,
        decoder_input_sequences,
        decoder_output_sequences,
        test_size=0.2,
        random_state=42,
    )

    return (
        encoder_input_train,
        encoder_input_val,
        decoder_input_train,
        decoder_input_val,
        decoder_output_train,
        decoder_output_val,
        num_encoder_tokens,
        num_decoder_tokens,
        src_tokenizer,
        tar_tokenizer,
    )
