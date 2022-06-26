# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch.optim

from models import EncoderDecoder,KoopmanOperator,Lusch
from data_generator import load_dataset,differential_dataset
from loss_functions import koopman_loss,prediction_loss
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
import numpy as np
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    koopman_dim = 64
    hidden_dim = [500]
    input_dim = 3
    delta_t = 0.01


    epochs = 300
    lr = 1e-3
    Sp = 72; horizon = 72; T = horizon
    batch_size = 512
    load_chkpt = False
    chkpt_filename = "fixed_matrix"
    start_epoch = 1
    device="cuda"


    model = Lusch(input_dim,koopman_dim,hidden_dim = hidden_dim,delta_t=delta_t,device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    if load_chkpt:
        print("LOAD CHECKPOINTS")

    X_train,X_test = load_dataset(chunk_size=1)
    X_train_recon = X_train[:,:328,:]; X_test_recon = X_test[:,:328,:]
    X_forecast_train = X_train[:,328:,:]; X_forecast_test = X_test[:,328:,:]
    train_dl = DataLoader(differential_dataset(X_train_recon,horizon),batch_size=batch_size)
    test_dl = DataLoader(differential_dataset(X_test_recon,horizon),batch_size=batch_size)

    model.mu = train_dl.dataset.mu.to(device)
    model.std = train_dl.dataset.std.to(device)

    save_every = 5

    for epoch in range(start_epoch,epochs):
        train_epoch_loss = []
        test_epoch_loss = []

        model.train()
        for x in tqdm(train_dl):
            optimizer.zero_grad()
            loss_train = koopman_loss(x.to(device), model, Sp=Sp, T=T,alpha=2)
            loss_train.backward()
            optimizer.step()
            train_epoch_loss.append(loss_train.item()*x.shape[0])

        with torch.no_grad():
            model.eval()
            for x in tqdm(test_dl):
                loss_test = koopman_loss(x.to(device), model, Sp=Sp, T=T,alpha=2)
                test_epoch_loss.append(loss_test.cpu().item() * x.shape[0])

            forecast_loss = prediction_loss(X_test_recon[:,[-1],:].to(device), X_forecast_test.to(device),model, Sp=Sp, T=T,alpha=2)

        if (epoch+1) % save_every == 0:
            torch.save({"model":model.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "start_epoch":(epoch+1)},chkpt_filename+"teacher_force.pth")


        print("\n","="*10,f" EPOCH {epoch} ","="*10)
        print("Prediction Loss: {:.4f}".format(forecast_loss))
        # print("Reconstruction Loss: {:.4f}".format(reconstruction_loss))
        print("TRAIN LOSS: ",np.mean(train_epoch_loss))
        print("TEST LOSS: ",np.mean(test_epoch_loss))

#* FIDDLE

print("COMPLETE NOW :)")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
