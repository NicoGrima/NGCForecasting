import torch


def train_lstm(epochs, model, train_dataloader, val_dataloader, optimizer, criterion, device):
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        # Training
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
        # Testing
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()

        print('Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}'
              .format(epoch + 1, epochs, train_loss / len(train_dataloader), test_loss / len(val_dataloader)))
    # Save model
    torch.save(model.state_dict(), 'cnn_lstm.pth')


def train_transfortmer(epochs, model, train_dataloader, test_dataloader, optimizer, criterion, target_num, device):
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        # Training
        for i, (enc_input, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()

            enc_input = enc_input.to(device)
            labels = labels.to(device)
            # The decoder input is set as the last target element in the enc input
            # + all but the last element of the label
            dec_input = torch.cat((enc_input[:, target_num, -1].unsqueeze(1), labels[:, :-1]), dim=1)

            outputs = model.forward(enc_input, dec_input, training=True)
            loss = criterion(outputs.squeeze(-1), labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        test_loss = 0.0
        model.eval()
        # Testing
        with torch.no_grad():
            for i, (enc_input, labels) in enumerate(test_dataloader):
                enc_input = enc_input.to(device)
                labels = labels.to(device)

                # Initialize the decoder input with a start token (here, we use the last target value of the enc input)
                token_input = enc_input[:, target_num, -1:]
                dec_input = token_input

                # Generate the output sequence iteratively
                outputs = None
                for t in range(labels.size(1)):
                    outputs = model.forward(enc_input, dec_input, training=False)
                    # Add output (t+1 prediction) to the decoder input, then repeat
                    dec_input = torch.cat((token_input, outputs.squeeze(2)), dim=1)

                # Calculate the loss
                loss = criterion(outputs.squeeze(-1), labels)
                test_loss += loss.item()

        print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'
              .format(epoch + 1, epochs, train_loss / len(train_dataloader), test_loss / len(test_dataloader)))
    # Save model
    torch.save(model.state_dict(), 'transformer.pth')


def train_model(train_dataloader, test_dataloader, epochs, optimizer, criterion, model,
                target_num, device, model_type: str):
    if model_type == 'LSTM':
        train_lstm(epochs, model, train_dataloader, test_dataloader, optimizer, criterion, device)
    elif model_type == 'Transformer':
        train_transfortmer(epochs, model, train_dataloader, test_dataloader, optimizer, criterion, target_num, device)
    else:
        raise ValueError('Model type not available. Possible selections: "LSTM" or "Transformer"')


def cross_train_model(fold_dataloaders, epochs, optimizer, criterion, model, target_num,
                      device, model_type: str):
    for fold, (train_dataloader, val_dataloader) in enumerate(fold_dataloaders):
        print(f'FOLD {fold + 1}')
        if model_type == 'LSTM':
            train_lstm(epochs, model, train_dataloader, val_dataloader, optimizer, criterion, device)
        elif model_type == 'Transformer':
            train_transfortmer(epochs, model, train_dataloader, val_dataloader, optimizer, criterion, target_num,
                               device)
        else:
            raise ValueError('Model type not available. Possible selections: "LSTM" or "Transformer"')
