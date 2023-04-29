import torch


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
