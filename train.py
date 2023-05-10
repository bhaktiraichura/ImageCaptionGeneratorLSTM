import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LanguageDataset
from encoder import Encoder
from decoder import Decoder
from evaluation import evaluate

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize dataset and dataloader
train_dataset = LanguageDataset("train_data.txt")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize encoder and decoder
encoder = Encoder(input_size=train_dataset.num_chars, hidden_size=256).to(device)
decoder = Decoder(output_size=train_dataset.num_chars, hidden_size=256).to(device)

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Set number of epochs and start training
num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0

    for batch_idx, (input_data, target_data) in enumerate(train_loader):
        # Move input and target data to GPU if available
        input_data = input_data.to(device)
        target_data = target_data.to(device)

        # Zero the gradients and forward pass
        optimizer.zero_grad()
        encoder_output = encoder(input_data)
        decoder_output = decoder(target_data, encoder_output)

        # Calculate loss and backpropagate
        loss = criterion(decoder_output.view(-1, train_dataset.num_chars), target_data.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Evaluate on validation set after each epoch
    val_loss, val_accuracy = evaluate(encoder, decoder, "val_data.txt", train_dataset.num_chars, device)
    print("Epoch [{}/{}], Loss: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.4f}".format(epoch+1, num_epochs, total_loss/len(train_loader), val_loss, val_accuracy))
    
# Save trained models
torch.save(encoder.state_dict(), "encoder.pth")
torch.save(decoder.state_dict(), "decoder.pth")
