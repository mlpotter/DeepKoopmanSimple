# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch.optim
import torch.nn.functional as F

from models import LorenzEmbedding,LorenzEmbeddingTrainer

from data_generator import load_dataset,differential_dataset
from loss_functions import koopman_loss,prediction_loss
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
import numpy as np
# Press the green button in the gutter to run the script.

class config(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    device = "cuda:0"

    config.state_dim = 3
    config.n_embd = 64
    config.layer_norm_epsilon = 1e-05
    config.embd_pdrop = 0.0
    config.device = device

    epochs = 300
    lr = 1e-3
    horizon = 328; T = 400; Trajectory_Size = 64;
    batch_size = 64
    load_chkpt = False
    chkpt_filename = "koopman_"+str(config.n_embd)
    start_epoch = 1


    X_train,X_test = load_dataset(chunk_size=1)
    X_train_recon = X_train[:,:horizon,:]; X_test_recon = X_test[:,:horizon,:]
    X_forecast_train = X_train[:,horizon:,:]; X_forecast_test = X_test[:,horizon:,:]
    train_dl = DataLoader(differential_dataset(X_train_recon,Trajectory_Size),batch_size=batch_size)
    test_dl = DataLoader(differential_dataset(X_test_recon,Trajectory_Size),batch_size=batch_size)

    save_every = 5

    trainer = LorenzEmbeddingTrainer(config)
    trainer.embedding_model.mu = train_dl.dataset.mu.to(device)
    trainer.embedding_model.std = train_dl.dataset.std.to(device)

    print(train_dl.dataset.mu)

    optimizer = torch.optim.Adam(trainer.parameters(), lr=lr*0.995 ** (start_epoch - 1), weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    print("STARTED SCRIPT")

    for epoch in range(start_epoch,epochs):
        train_epoch_loss = []
        test_epoch_loss = []

        trainer.train();
        for x in tqdm(train_dl):
            optimizer.zero_grad()
            [loss_train,_] = trainer(x.to(device)) #, encoder, decoder, koopman, Sp=Sp, T=T)
            loss_train = loss_train.sum()
            loss_train.backward()
            torch.nn.utils.clip_grad_norm_(trainer.parameters(), 0.1)
            optimizer.step()
            train_epoch_loss.append(loss_train.item()*x.shape[0])

        scheduler.step()
        with torch.no_grad():
            trainer.eval()
            for x in tqdm(test_dl):
                # loss_test = koopman_loss(x, encoder, decoder, koopman, Sp=Sp, T=T)
                [loss_test,_,_] = trainer.evaluate(x.to(device))
                test_epoch_loss.append(loss_test.item() * x.shape[0])

            forecast = trainer.predict_ahead(X_test_recon[:,[-1],:],T-horizon)
            forecast_loss = F.mse_loss(forecast,X_forecast_test.to(device)).cpu()
        # reconstruction_loss,forecast_loss = prediction_loss(X_test_recon.to(device), X_forecast_test.to(device),encoder, decoder, koopman, Sp=Sp, T=T)

        if (epoch+1) % save_every == 0:
            # torch.save(trainer.state_dict(),chkpt_filename+"koopman.pth")
            # torch.save(optimizer.state_dict(),chkpt_filename+"optimizer.pth")
            print("Save Chkpt")
            torch.save({"model":trainer.state_dict(),
             "optimizer":optimizer.state_dict(),
             "scheduler":scheduler.state_dict(),
             "start_epoch":(epoch+1)},chkpt_filename+"_physical.pth")



        print("\n","="*10,f" EPOCH {epoch} ","="*10)
        print("Prediction Loss: {:.4f}".format(forecast_loss))
        # print("Reconstruction Loss: {:.4f}".format(reconstruction_loss))
        print("TRAIN LOSS: ",np.mean(train_epoch_loss))
        print("TEST LOSS: ",np.mean(test_epoch_loss))

#* FIDDLE

print("COMPLETE NOW :)")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
