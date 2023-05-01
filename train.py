import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold


def train_model(train_dataloader, test_dataloader, epochs, optimizer, criterion, model, device):
    for epoch in range(epochs):
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        test_loss = 0.0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()

        print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'
              .format(epoch + 1, epochs, train_loss / len(train_dataloader), test_loss / len(test_dataloader)))

    # Save model
    torch.save(model.state_dict(), 'cnn_lstm.pth')


def cross_train_model(fold_dataloaders, epochs, optimizer, criterion, model, device):
    for fold, (train_dataloader, val_dataloader) in enumerate(fold_dataloaders):
        print(f'FOLD {fold + 1}')

        for epoch in range(epochs):
            train_loss = 0.0
            model.train()

            for i, (inputs, labels) in enumerate(train_dataloader):
                optimizer.zero_grad()

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            test_loss = 0.0
            model.eval()
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(val_dataloader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model.forward(inputs)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()

            print('Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}'
                  .format(epoch + 1, epochs, train_loss / len(train_dataloader), test_loss / len(val_dataloader)))

