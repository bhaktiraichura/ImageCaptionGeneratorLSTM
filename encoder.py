from tensorflow.keras.layers import LSTM, Input
from tensorflow.keras import Model


def build_encoder(num_encoder_tokens, latent_dim):
    # Define input layer
    encoder_inputs = Input(shape=(None,))

    # Define LSTM layer
    encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)

    # Connect input to LSTM
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

    # Discard outputs and only keep states
    encoder_states = [state_h, state_c]

    # Define encoder model
    encoder_model = Model(encoder_inputs, encoder_states)

    return encoder_inputs, encoder_lstm, encoder_model
