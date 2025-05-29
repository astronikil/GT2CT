import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, X in enumerate(dataloader):
        
        coord, target = X['coord'], X['subclass']
        target = target #.to(device)
        coord = coord #.to(device)

        pred = model(coord)

        loss = loss_fn( pred, target )
        loss.backward()
        
        clip_value = 0.5
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)

        optimizer.step()
        optimizer.zero_grad()
        
        
def topk_accuracy(output, target, maxk=5):
    """
    Calculates the top-5 accuracy for a batch of predictions.

    Args:
        output (torch.Tensor): Output tensor of the classifier (e.g., logits or probabilities).
                               Shape: (batch_size, num_classes)
        target (torch.Tensor): Ground truth labels.
                               Shape: (batch_size,)

    Returns:
        float: The top-5 accuracy as a percentage (0.0 to 100.0).
    """
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:maxk].reshape(-1).float().sum(0)
    return correct_k.item()

def test_loop(dataloader, model, loss_fn, maxk=3):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X in dataloader:
            coord, target = X['coord'], X['subclass']
            target = target #.to(device)
            coord = coord #.to(device)

            pred = model(coord)

            ls = loss_fn( pred, target )
            test_loss += ls.item()
            arg = torch.argsort(pred)
            correct += topk_accuracy(pred, target, maxk=maxk)

    test_loss /= num_batches
    correct /= size
    print(f"Top-{maxk:>1d} Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return test_loss
