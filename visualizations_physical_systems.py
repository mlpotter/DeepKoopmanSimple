import torch.optim

from models import EncoderDecoder,KoopmanOperator
from data_generator import load_dataset,differential_dataset
from loss_functions import koopman_loss,prediction_loss
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from models import LorenzEmbedding,LorenzEmbeddingTrainer
import torch.nn.functional as F

class config(object):
    def __init__(self):
        pass
# Press the green button in the gutter to run the script.

if __name__ == '__main__':

    device = "cuda:0"

    config.state_dim = 3
    config.n_embd = 64
    config.layer_norm_epsilon = 1e-05
    config.embd_pdrop = 0.0
    config.device = device
    LorenzEmbedding(config)

    epochs = 300
    lr = 1e-3
    horizon = 328;
    T = 400;
    batch_size = 512
    load_chkpt = True
    chkpt_filename = "koopman_"+str(config.n_embd)
    start_epoch = 1

    X_train, X_test = load_dataset(chunk_size=1)
    X_train_recon = X_train[:, :horizon, :];
    X_test_recon = X_test[:, :horizon, :]
    X_forecast_train = X_train[:, horizon:, :];
    X_forecast_test = X_test[:, horizon:, :]
    train_dl = DataLoader(differential_dataset(X_train_recon, horizon), batch_size=batch_size)
    test_dl = DataLoader(differential_dataset(X_test_recon, horizon), batch_size=batch_size)


    trainer = LorenzEmbeddingTrainer(config)
    # torch.load(load_chkpt)
    if load_chkpt:
        dict = torch.load(chkpt_filename+"_physical.pth")
        trainer.load_state_dict(dict['model'])

    trainer.eval()
    forecast = trainer.predict_ahead(X_test_recon[:, [-1], :], T - horizon)
    forecast_loss = F.mse_loss(forecast, X_forecast_test.to(device)).cpu()
    print("Prediction Error: ",forecast_loss)
    n = 15
    with torch.no_grad():
        with torch.inference_mode():
            trainer.eval()

            x_recon = X_test_recon[[n],:,:].to(device)
            x_ahead = X_forecast_test[[n],:,:].to(device)
            [_,x_recon_hat] = trainer.embedding_model.forward(x_recon)

            x_ahead_hat = trainer.predict_ahead(x_recon[:,[-1],:],x_ahead.shape[1])


        mpl.use('Qt5Agg')
        plt.figure(figsize=(20, 10))
        #     for i in range(3):
        plt.plot(np.arange(x_recon.shape[1]), x_recon_hat.squeeze().cpu())
        plt.plot(np.arange(X_test.shape[1]), X_test[n, :, :].cpu(), '--')
        plt.plot(x_recon.shape[1] + np.arange(x_ahead.shape[1]), x_ahead_hat[0, :, :].cpu(), 'r.')

        plt.xlabel("Time (n)", fontsize=20)
        plt.ylabel("State", fontsize=20)
        plt.legend(["x", "y", "z", "$x_{reconstructed}$", "$y_{reconstructed}$", "$z_{reconstructed}$", "Prediction"],
                   fontsize=20)
        plt.show()



        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(X_test[n, :, 0].cpu(), X_test[n, :, 1].cpu(), X_test[n, :, 2].cpu(), 'k-')  # c=np.linspace(0,1,Time_Length))
        ax.plot3D(x_recon_hat[0, :, 0].cpu(), x_recon_hat[0, :, 1].cpu(), x_recon_hat[0, :, 2].cpu(), 'b*')
        ax.plot3D(x_ahead_hat[0, :, 0].cpu(), x_ahead_hat[0, :, 1].cpu(), x_ahead_hat[0, :, 2].cpu(), 'rx')
        ax.set_xlabel('$X$', fontsize=20)
        ax.set_ylabel('$Y$', fontsize=20)
        ax.set_zlabel(r'$Z$', fontsize=20)
        plt.legend(["Actual", "Reconstruction", "Prediction"])
        plt.show()