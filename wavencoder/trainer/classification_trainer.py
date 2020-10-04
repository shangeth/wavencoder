import torch
import torch.nn as nn
import numpy as np
import time
import wavencoder
from tqdm import tqdm

def train(model, trainloader, valloader, n_epochs, 
    optimizer=None,
    lr=1e-3,
    scheduler=None,
    criterion=nn.CrossEntropyLoss(),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_path="trained_model.pt"):

    model.to(device)
    criterion.to(device)
    if optimizer == None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_loss_min = np.inf

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    for i_epoch in range(n_epochs):
        epoch_train_loss = 0
        epoch_train_acc = 0
        model.train()
        for batch in tqdm(trainloader, desc="Train"):
            batch_x, batch_y = batch
            batch_x, batch_y = batch_x.float().to(device), batch_y.long().view(-1).to(device)
            y_hat = model(batch_x)
            loss = criterion(y_hat, batch_y)
            epoch_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, max_indices = torch.max(y_hat, 1)
            epoch_train_acc += (max_indices == batch_y).sum().data.cpu().numpy()/max_indices.size()[0]
    
        epoch_train_loss = epoch_train_loss/len(trainloader)
        epoch_train_acc = epoch_train_acc/len(trainloader)
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)


        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0
            epoch_val_acc = 0
            for batch in tqdm(valloader, desc="Val  "):
                batch_x, batch_y = batch
                batch_x, batch_y = batch_x.float().to(device), batch_y.long().view(-1).to(device)
                y_hat = model(batch_x)
                loss = criterion(y_hat, batch_y)
                epoch_val_loss += loss.item()

                _, max_indices = torch.max(y_hat, 1)
                epoch_val_acc += (max_indices == batch_y).sum().data.cpu().numpy()/max_indices.size()[0]

            epoch_val_loss = epoch_val_loss/len(valloader)
            epoch_val_acc = epoch_val_acc/len(valloader)
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            if epoch_val_loss < val_loss_min:
                torch.save(model.state_dict(), save_path)
                tqdm.write(f'Validation loss reduced from {val_loss_min:.6f} to {epoch_val_loss:.6f}, saving model at {save_path} ...')
                val_loss_min = epoch_val_loss

        if scheduler:
            scheduler.step()
            tqdm.write(f'Updating lr to {scheduler.get_last_lr()}')

        tqdm.write(f'Epoch : {i_epoch+1:02}\nTrain Loss = {epoch_train_loss:.4f}\tTrain Acc = {epoch_train_acc}\n  Val Loss = {epoch_val_loss:.4f}\t  Val Acc = {epoch_val_acc}\n')

    loss_dict = {"train_losses" : train_losses,
                 "val_losses" : val_losses,
                 "train_accuracies" : train_accuracies,
                 "val_Accuracies" : val_accuracies}

    return model, loss_dict

def test_evaluate_classifier(model, testloader, 
    criterion=nn.CrossEntropyLoss(),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.to(device).eval()
    with torch.no_grad():
        test_val_loss = 0
        test_val_acc = 0
        for batch in tqdm(testloader):
            batch_x, batch_y = batch
            batch_x, batch_y = batch_x.float().to(device), batch_y.long().view(-1).to(device)
            y_hat = model(batch_x)
            loss = criterion(y_hat, batch_y)
            test_val_loss += loss.item()

            _, max_indices = torch.max(y_hat, 1)
            test_val_acc += (max_indices == batch_y).sum().data.cpu().numpy()/max_indices.size()[0]

        test_val_loss = test_val_loss/len(testloader)
        test_val_acc = test_val_acc/len(testloader)

        loss_dict = {"test_loss" : test_val_loss,
                     "test_acc" : test_val_acc}
    return loss_dict

def test_predict_classifier(model, testloader,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.to(device).eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(testloader):
            batch_x, _ = batch
            batch_x = batch_x.float().to(device)
            y_hat = model(batch_x)

            _, max_indices = torch.max(y_hat, 1)
            predictions += max_indices.cpu().detach().view(-1).numpy().tolist()

        loss_dict = {"test_predictions" : predictions}
    return loss_dict