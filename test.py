import torch
from torch.utils.data import DataLoader
from dataset import Seq2SeqDataset
from encoder import Encoder
from decoder import Decoder

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
test_dataset = Seq2SeqDataset('test.txt')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Load models
encoder = Encoder(input_size=128, hidden_size=256).to(device)
decoder = Decoder(output_size=128, hidden_size=256).to(device)

encoder.load_state_dict(torch.load('encoder.pth'))
decoder.load_state_dict(torch.load('decoder.pth'))

encoder.eval()
decoder.eval()

# Test loop
total_loss = 0
with torch.no_grad():
    for i, (input_seq, target_seq) in enumerate(test_loader):
        # Move data to device
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        # Encode input sequence
        encoder_hidden = encoder.init_hidden()
        encoder_outputs, encoder_hidden = encoder(input_seq, encoder_hidden)

        # Initialize decoder input
        decoder_input = torch.tensor([[0]], device=device)

        # Initialize decoder hidden state with encoder final hidden state
        decoder_hidden = encoder_hidden

        # Decode sequence
        decoded_seq = []
        for t in range(target_seq.size(1)):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoded_seq.append(decoder_output.argmax(1).item())
            decoder_input = target_seq[:, t].unsqueeze(1)

        # Calculate loss
        loss = torch.nn.functional.cross_entropy(torch.tensor(decoded_seq, device=device), target_seq.squeeze(0))
        total_loss += loss.item()

        # Print decoded sequence
        print(f"Input Sequence: {input_seq.tolist()[0]}")
        print(f"Target Sequence: {target_seq.tolist()[0]}")
        print(f"Decoded Sequence: {decoded_seq}")
        print(f"Loss: {loss.item()}")

# Print average loss
print(f"Average Loss: {total_loss / len(test_loader)}")
