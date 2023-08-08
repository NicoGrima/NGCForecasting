import torch


def train_lstm(epochs, model, train_dataloader, val_dataloader, optimizer, criterion, save_file, device, neptune):
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
        # neptune["train/loss"].append(test_loss)

    # Save model
    torch.save(model.state_dict(), save_file)


def train_transfortmer(epochs, model, train_dataloader, test_dataloader, optimizer, criterion, target_num, save_file,
                       step=10, device='cpu', neptune=None, scheduler=None):
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
            # Let us hold off on the autoregressive behavior given the need for quick results in our task
            # last_enc = enc_input[:, target_num, -step:]
            # first_labels = labels[:, :-step]
            # dec_input = torch.cat((last_enc, first_labels), dim=1).unsqueeze(1)
            dec_input = enc_input[:, :, -labels.shape[1]:]  # we will use the last elements of the encoder input as our decoder input
            # dec_input = enc_input[:, :, -1]

            outputs = model.forward(enc_input, dec_input)
            loss = criterion(outputs.squeeze(-1), labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        scheduler.step()

        test_loss = 0.0
        model.eval()
        # Testing
        with torch.no_grad():
            for i, (enc_input, labels) in enumerate(test_dataloader):
                enc_input = enc_input.to(device)
                labels = labels.to(device)

                # Initialize the decoder input with a start token (here, we use the last 10 target value of the enc input)
                # token_input = enc_input[:, target_num, -step:]
                # token_input = token_input.unsqueeze(1)
                # dec_input = token_input

                dec_input = enc_input[:, :, -labels.shape[1]:]
                # dec_input = enc_input[:, :, -1]
                outputs = model.forward(enc_input, dec_input)

                # Generate the output sequence iteratively (ignore autoregression for now)
                # outputs = None
                # for t in range(0, labels.size(1), step):
                #     outputs = model.forward(enc_input, dec_input)
                #     # Add output (t+1 prediction) to the decoder input, then repeat
                #     out = outputs.permute(0, 2, 1)
                #     dec_input = torch.cat((dec_input, out[:, :, -step:]), dim=2)

                # Calculate the loss
                loss = criterion(outputs.squeeze(-1), labels)
                test_loss += loss.item()

        print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'
              .format(epoch + 1, epochs, train_loss / len(train_dataloader), test_loss / len(test_dataloader)))
        # neptune["train/loss"].append(test_loss)  # need to change

    # Save model
    torch.save(model.state_dict(), save_file)


def train_model(train_dataloader, test_dataloader, epochs, optimizer, criterion, model, target_num,
                save_file, step, device, model_type: str, neptune, scheduler):
    if model_type == 'LSTM' or model_type == 'vLSTM':
        train_lstm(epochs, model, train_dataloader, test_dataloader, optimizer, criterion, save_file, device,
                   neptune)
    elif model_type == 'Transformer':
        train_transfortmer(epochs, model, train_dataloader, test_dataloader, optimizer, criterion, target_num,
                           save_file, step, device, neptune, scheduler)
    else:
        raise ValueError('Model type not available. Possible selections: "LSTM", "Transformer", or "vLSTM"')


def cross_train_model(fold_dataloaders, epochs, optimizer, criterion, model, target_num, save_file, step,
                      device, model_type: str, neptune):
    for fold, (train_dataloader, val_dataloader) in enumerate(fold_dataloaders):
        print(f'FOLD {fold + 1}')
        if model_type == 'LSTM' or model_type == 'vLSTM':
            train_lstm(epochs, model, train_dataloader, val_dataloader, optimizer, criterion, save_file, device,
                       neptune)
        elif model_type == 'Transformer':
            train_transfortmer(epochs, model, train_dataloader, val_dataloader, optimizer, criterion, target_num,
                               save_file, step, device, neptune)
        else:
            raise ValueError('Model type not available. Possible selections: "LSTM", "Transformer", or "vLSTM"')
