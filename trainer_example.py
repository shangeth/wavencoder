from wavencoder.models import Wav2Vec, LSTM_Attn_Classifier
from wavencoder.trainer import train, test_evaluate_classifier, test_predict_classifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

# --------------------------------
model = nn.Sequential(
    Wav2Vec(pretrained=False),
    LSTM_Attn_Classifier(512, 64, 2)
)

print(model)

# --------------------------------
X_train = torch.randn(10, 1, 16000)
y_train = torch.empty(10, 1).random_(1)

X_val = torch.randn(5, 1, 16000)
y_val = torch.empty(5, 1).random_(1)

X_test = torch.randn(10, 1, 16000)
y_test = torch.empty(10, 1).random_(1)

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
test_ds = TensorDataset(X_test, y_test)

trainloader = torch.utils.data.DataLoader(train_ds, batch_size=10)
valloader = torch.utils.data.DataLoader(val_ds, batch_size=5)
testloader = torch.utils.data.DataLoader(test_ds, batch_size=5)

# -------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model, train_dict = train(model, trainloader, valloader, n_epochs=5, optimizer=optimizer)
# test_dict = test_evaluate_classifier(model, testloader)
test_dict = test_predict_classifier(model, testloader)
print(test_dict)
