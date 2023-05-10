import torch
import torch.nn.functional as F

def evaluate(model, data_loader, device):
    model.eval()

    with torch.no_grad():
        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        for batch_idx, (input_seq, target_seq) in enumerate(data_loader):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            # Forward pass
            output_seq = model(input_seq)

            # Compute loss
            loss = F.cross_entropy(output_seq.view(-1, output_seq.size(-1)), target_seq.view(-1))
            total_loss += loss.item()

            # Compute accuracy
            pred = output_seq.argmax(dim=-1)
            accuracy = torch.mean((pred == target_seq).float())
            total_accuracy += accuracy.item()

            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        print('Evaluation - loss: {:.4f}, accuracy: {:.4f}'.format(avg_loss, avg_accuracy))

        return avg_loss, avg_accuracy
