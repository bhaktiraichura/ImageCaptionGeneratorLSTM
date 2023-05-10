import torch
from torch.utils.data import DataLoader
from dataset import TextDataset
from encoder import Encoder
from decoder import Decoder

# Define the path to the saved model
MODEL_PATH = "model.pt"

# Define the batch size to use for testing
BATCH_SIZE = 32

# Define the device to use for testing (CPU or GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the test dataset
test_dataset = TextDataset("test.txt")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Load the saved model
encoder = Encoder(input_size=len(test_dataset.vocab), hidden_size=256)
decoder = Decoder(output_size=len(test_dataset.vocab), hidden_size=256)
model = torch.nn.Sequential(encoder, decoder).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# Evaluate the model on the test dataset
model.eval()
with torch.no_grad():
    total_loss = 0
    total_correct = 0
    total_predictions = 0
    for input_seq, target_seq in test_loader:
        input_seq = input_seq.to(DEVICE)
        target_seq = target_seq.to(DEVICE)
        
        encoder_hidden = encoder.init_hidden(BATCH_SIZE)
        encoder_outputs, encoder_hidden = encoder(input_seq, encoder_hidden)
        decoder_hidden = encoder_hidden
        decoder_input = target_seq[:, 0].unsqueeze(1)
        
        loss = 0
        correct = 0
        predictions = 0
        for t in range(1, target_seq.size(1)):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += torch.nn.functional.cross_entropy(decoder_output, target_seq[:, t])
            _, topi = decoder_output.topk(1)
            correct += (topi.squeeze() == target_seq[:, t]).sum().item()
            predictions += target_seq.size(0)
            decoder_input = topi.detach()
        
        total_loss += loss.item() * target_seq.size(0)
        total_correct += correct
        total_predictions += predictions
    
    avg_loss = total_loss / len(test_dataset)
    accuracy = total_correct / total_predictions
    f1_score = 2 * (accuracy * (1 - accuracy)) / (accuracy + (1 - accuracy))
    print("Test Loss: {:.4f}, Accuracy: {:.4f}, F1 Score: {:.4f}".format(avg_loss, accuracy, f1_score))
