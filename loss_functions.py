import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def koopman_loss(x,model,Sp,T,alpha=2):
    # Sp < T

    encoder_x = model.embed(x)

    reconstruction_loss = F.mse_loss(model.recover(encoder_x),x)

    MAX_T = max(Sp,T)
    koopman_stepped = model.koopman_operator(encoder_x[:,:MAX_T,:]) #koopmanoperator.multiple_forward(encoder_x[:,[0],:],MAX_T)


    pred_loss = F.mse_loss(x[:,1:Sp,:],model.recover(koopman_stepped[:,:(Sp-1),:]))
    # pred_loss = torch.mean(torch.stack([F.mse_loss(x[:,[s],:],decoder(koopman_stepped[:,[s-1],:])) for s in range(1,Sp-1)]))

    lin_loss = F.mse_loss(encoder_x[:,1:T,:],koopman_stepped[:,:(T-1),:])
    # lin_loss = torch.mean(torch.stack([F.mse_loss(encoder_x[:,[t],:],koopman_stepped[:,[t-1],:]) for t in range(1,T-1)]))

    loss = alpha*(pred_loss + reconstruction_loss) + lin_loss
    return loss

def prediction_loss(x_recon,x_ahead,model,Sp,T,alpha=.5):
    # Sp < T
    with torch.inference_mode():
        model.eval()
        X = torch.concat((x_recon[:,[-1],:],x_ahead[:,:-1,:]),1)
        Y = model.koopman_operator(model.embed(X))
        prediction_loss = F.mse_loss(x_ahead,model.recover(Y))

    return prediction_loss

