import torch.optim

from models import EncoderDecoder,KoopmanOperator,Lusch
from data_generator import load_dataset,differential_dataset
from loss_functions import koopman_loss,prediction_loss
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    koopman_dim = 64
    hidden_dim = [500]
    input_dim = 3
    delta_t = 0.01

    epochs = 300
    lr = 1e-3
    Sp = 72;
    horizon = 72;
    T = horizon
    batch_size = 128
    load_chkpt = True
    chkpt_filename = "fixed_matrix"
    start_epoch = 1
    device = "cuda"

    model = Lusch(input_dim, koopman_dim, hidden_dim=hidden_dim, delta_t=delta_t, device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train, X_test = load_dataset(chunk_size=1)
    X_train_recon = X_train[:, :328, :];
    X_test_recon = X_test[:, :328, :].to(device)
    X_forecast_train = X_train[:, 328:, :]
    X_forecast_test = X_test[:, 328:, :].to(device);
    train_dl = DataLoader(differential_dataset(X_train_recon, horizon), batch_size=batch_size)
    test_dl = DataLoader(differential_dataset(X_test_recon, horizon), batch_size=batch_size)

    model.mu = train_dl.dataset.mu.to(device)
    model.std = train_dl.dataset.std.to(device)

    save_every = 5

    if load_chkpt:
        print("LOAD CHECKPOINTS")
        state_dicts = torch.load(chkpt_filename+"teacher_force_best.pth")
        model.load_state_dict(state_dicts['model'])


    with torch.inference_mode():
        model.eval()
        n = 10
        x_recon_hat = model.recover(model.embed(X_test_recon)).cpu()
        X = torch.concat((X_test_recon[:,[-1],:],X_forecast_test[:,:-1,:]),1)
        x_ahead_hat = model.recover(model.koopman_operator(model.embed(X))).cpu()

        x_recon_hat = model.recover(model.koopman_operator(model.embed(X_test_recon))).cpu()



        mpl.use('Qt5Agg')
        plt.figure(figsize=(20, 10))
        #     for i in range(3):
        plt.plot(np.arange(328), x_recon_hat[n,:,:])
        plt.plot(np.arange(400), X_test[n, :, :], '--')
        plt.plot(328 + np.arange(72), x_ahead_hat[n, :, :].cpu(), 'r.')

        plt.xlabel("Time (n)", fontsize=20)
        plt.ylabel("State", fontsize=20)
        plt.legend(["x", "y", "z", "$x_{reconstructed}$", "$y_{reconstructed}$", "$z_{reconstructed}$", "Prediction"],
                   fontsize=20)
        plt.show()



        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(X_test[n, :, 0].cpu(), X_test[n, :, 1].cpu(), X_test[n, :, 2].cpu(), 'k-')  # c=np.linspace(0,1,Time_Length))
        ax.plot3D(x_recon_hat[n, :, 0].cpu(), x_recon_hat[n, :, 1].cpu(), x_recon_hat[n, :, 2].cpu(), 'b*')
        ax.plot3D(x_ahead_hat[n, :, 0].cpu(), x_ahead_hat[n, :, 1].cpu(), x_ahead_hat[n, :, 2].cpu(), 'rx')
        ax.set_xlabel('$X$', fontsize=20)
        ax.set_ylabel('$Y$', fontsize=20)
        ax.set_zlabel(r'$Z$', fontsize=20)
        plt.legend(["Actual", "Reconstruction", "Prediction"])
        plt.show()