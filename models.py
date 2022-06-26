import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from torch.autograd import Variable
class EncoderDecoder(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim = []):
        super(EncoderDecoder,self).__init__()

        self.hidden_layers = len(hidden_dim)
        self.output_dim = output_dim
        if len(hidden_dim) == 0:
            self.inp = nn.Linear(input_dim,output_dim)

        elif len(hidden_dim) == 1:
            self.inp = nn.Linear(input_dim,hidden_dim[0])
            self.hidden = nn.ModuleList([nn.Linear(hidden_dim[0],hidden_dim[0])])
            self.out = nn.Linear(hidden_dim[0],output_dim)
        else:
            self.inp = nn.Linear(input_dim,hidden_dim[0])
            self.hidden = nn.ModuleList([nn.Linear(hidden_dim[i],hidden_dim[i+1]) for i in range(0,len(hidden_dim)-1)])
            self.out = nn.Linear(hidden_dim[-1],output_dim)

    def forward(self,x):

        x = torch.tanh(self.inp(x))

        if self.hidden_layers > 0:
            for layer in self.hidden:
                x = torch.tanh(layer(x))

        x = self.out(x)

        return x

class KoopmanOperator(nn.Module):
    def __init__(self,koopman_dim,delta_t,device="cpu"):
        super(KoopmanOperator,self).__init__()

        self.koopman_dim = koopman_dim
        self.num_eigenvalues = int(koopman_dim/2)
        self.delta_t = delta_t
        self.parameterization = nn.Sequential(
            nn.Linear(self.koopman_dim,self.num_eigenvalues*2),
            nn.Tanh(),
            nn.Linear(self.num_eigenvalues*2,self.num_eigenvalues*2)
        )
        self.device = device

    def forward(self,x):
        # x is B x 1 x Latent
        # it is the one because only initial point (T=1)

        # mu is B x T x Latent/2
        # omega is B x T x Latent/2

        Y = Variable(torch.zeros(x.shape[0],x.shape[1],self.koopman_dim)).to(self.device)
        y = x[:,0,:]
        for t in range(x.shape[1]):
            mu,omega = torch.unbind(self.parameterization(y).reshape(-1,self.num_eigenvalues,2),-1)

            # K is B x Latent x Latent
            # K = torch.zeros((x.shape[0],self.latent_dim,self.latent_dim))

            # B x Koopmandim/2
            exp = torch.exp(self.delta_t * mu)

            # B x T x Latent/2
            cos = torch.cos(self.delta_t * omega)
            sin = torch.sin(self.delta_t * omega)


            K = Variable(torch.zeros(x.shape[0],self.koopman_dim,self.koopman_dim)).to(self.device)

            for i in range(0,self.koopman_dim,2):
                #for j in range(i,i+2):
                index = (i)//2

                K[:, i + 0, i + 0] = cos[:,index] *  exp[:,index]
                K[:, i + 0, i + 1] = -sin[:,index] * exp[:,index]
                K[:, i + 1, i + 0] = sin[:,index]  * exp[:,index]
                K[:, i + 1, i + 1] = -cos[:,index] * exp[:,index]

            y = torch.matmul(K,y.unsqueeze(-1)).squeeze(-1)
            Y[:,t,:] = y
            # x = torch.matmul(K, x.unsqueeze(-1)).squeeze(-1)
        return Y


class Lusch(nn.Module):
    def __init__(self,input_dim,koopman_dim,hidden_dim = [],delta_t=0.01,device="cpu"):
        super(Lusch,self).__init__()

        self.encoder = EncoderDecoder(input_dim, koopman_dim, hidden_dim)
        self.decoder = EncoderDecoder(koopman_dim, input_dim, hidden_dim)
        self.koopman = KoopmanOperator(koopman_dim,delta_t,device)

        self.device = device
        self.delta_t = delta_t

        # Normalization occurs inside the model
        self.register_buffer('mu', torch.zeros((input_dim,)))
        self.register_buffer('std', torch.ones((input_dim,)))

    def forward(self,x):
        x = self._normalize(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self._unnormalize(x)
        return x

    def embed(self,x):
        x = self._normalize(x)
        x = self.encoder(x)
        return x

    def recover(self,x):
        x = self.decoder(x)
        x = self._unnormalize(x)
        return x

    def koopman_operator(self,x):
        return self.koopman(x)

    def _normalize(self, x):
        return (x - self.mu.unsqueeze(0).unsqueeze(0))/self.std.unsqueeze(0).unsqueeze(0)

    def _unnormalize(self, x):
        return self.std.unsqueeze(0).unsqueeze(0)*x + self.mu.unsqueeze(0).unsqueeze(0)




class LorenzEmbedding(nn.Module):
    """Embedding Koopman model for the Lorenz ODE system
    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
    """
    model_name = "embedding_lorenz"

    def __init__(self, config):
        """Constructor method
        """
        super().__init__()

        # hidden_states = int(abs(config.state_dims[0] - config.n_embd)/2) + 1
        hidden_states = 500

        self.observableNet = nn.Sequential(
            nn.Linear(config.state_dim, hidden_states),
            nn.ReLU(),
            nn.Linear(hidden_states, config.n_embd),
            nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon),
            nn.Dropout(config.embd_pdrop)
        )

        self.recoveryNet = nn.Sequential(
            nn.Linear(config.n_embd, hidden_states),
            nn.ReLU(),
            nn.Linear(hidden_states, config.state_dim)
        )
        # Learned Koopman operator
        self.obsdim = config.n_embd
        self.kMatrixDiag = nn.Parameter(torch.linspace(1, 0, config.n_embd))

        # Off-diagonal indices
        xidx = []
        yidx = []
        for i in range(1, 3):
            yidx.append(np.arange(i, config.n_embd))
            xidx.append(np.arange(0, config.n_embd-i))

        self.xidx = torch.LongTensor(np.concatenate(xidx))
        self.yidx = torch.LongTensor(np.concatenate(yidx))
        self.kMatrixUT = nn.Parameter(0.1*torch.rand(self.xidx.size(0)))

        # Normalization occurs inside the model
        self.register_buffer('mu', torch.tensor([0., 0., 0.]))
        self.register_buffer('std', torch.tensor([1., 1., 1.]))

    def forward(self, x):
        """Forward pass
        Args:
            x (Tensor): [B, 3] Input feature tensor
        Returns:
            TensorTuple: Tuple containing:
                | (Tensor): [B, config.n_embd] Koopman observables
                | (Tensor): [B, 3] Recovered feature tensor
        """
        # Encode
        x = self._normalize(x)
        g = self.observableNet(x)
        # Decode
        out = self.recoveryNet(g)
        xhat = self._unnormalize(out)
        return g, xhat

    def embed(self, x):
        """Embeds tensor of state variables to Koopman observables
        Args:
            x (Tensor): [B, 3] Input feature tensor
        Returns:
            Tensor: [B, config.n_embd] Koopman observables
        """
        x = self._normalize(x)
        g = self.observableNet(x)
        return g

    def recover(self, g):
        """Recovers feature tensor from Koopman observables
        Args:
            g (Tensor): [B, config.n_embd] Koopman observables
        Returns:
            Tensor: [B, 3] Physical feature tensor
        """
        out = self.recoveryNet(g)
        x = self._unnormalize(out)
        return x

    def koopmanOperation(self, g):
        """Applies the learned Koopman operator on the given observables
        Args:
            g (Tensor): [B, config.n_embd] Koopman observables
        Returns:
            (Tensor): [B, config.n_embd] Koopman observables at the next time-step
        """
        # Koopman operator
        kMatrix = Variable(torch.zeros(self.obsdim, self.obsdim)).to(self.kMatrixUT.device)
        # Populate the off diagonal terms
        kMatrix[self.xidx, self.yidx] = self.kMatrixUT
        kMatrix[self.yidx, self.xidx] = -self.kMatrixUT

        # Populate the diagonal
        ind = np.diag_indices(kMatrix.shape[0])
        kMatrix[ind[0], ind[1]] = self.kMatrixDiag

        # Apply Koopman operation
        gnext = torch.bmm(kMatrix.expand(g.size(0), kMatrix.size(0), kMatrix.size(0)), g.unsqueeze(-1))
        self.kMatrix = kMatrix
        return gnext.squeeze(-1) # Squeeze empty dim from bmm

    @property
    def koopmanOperator(self, requires_grad =True):
        """Current Koopman operator
        Args:
            requires_grad (bool, optional): If to return with gradient storage. Defaults to True
        Returns:
            (Tensor): Full Koopman operator tensor
        """
        if not requires_grad:
            return self.kMatrix.detach()
        else:
            return self.kMatrix

    def _normalize(self, x):
        return (x - self.mu.unsqueeze(0))/self.std.unsqueeze(0)

    def _unnormalize(self, x):
        return self.std.unsqueeze(0)*x + self.mu.unsqueeze(0)

    @property
    def koopmanDiag(self):
        return self.kMatrixDiag


class LorenzEmbeddingTrainer(nn.Module):
    """Training head for the Lorenz embedding model
    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
    """

    def __init__(self, config):
        """Constructor method
        """
        super().__init__()
        self.embedding_model = LorenzEmbedding(config).to(config.device)
        self.device = config.device

    def forward(self, states):
        """Trains model for a single epoch
        Args:
            states (Tensor): [B, T, 3] Time-series feature tensor
        Returns:
            FloatTuple: Tuple containing:

                | (float): Koopman based loss of current epoch
                | (float): Reconstruction loss
        """
        self.embedding_model.train()


        loss_reconstruct = 0
        mseLoss = nn.MSELoss()

        xin0 = states[:, 0].to( self.device )  # Time-step

        # Model forward for initial time-step
        g0, xRec0 = self.embedding_model(xin0)
        loss = (1e4) * mseLoss(xin0, xRec0)
        loss_reconstruct = loss_reconstruct + mseLoss(xin0, xRec0).detach()

        g1_old = g0
        # Loop through time-series
        for t0 in range(1, states.shape[1]):
            xin0 = states[:, t0, :].to(self.device)  # Next time-step
            _, xRec1 = self.embedding_model(xin0)

            g1Pred = self.embedding_model.koopmanOperation(g1_old)
            xgRec1 = self.embedding_model.recover(g1Pred)

            loss = loss + mseLoss(xgRec1, xin0) + (1e4) * mseLoss(xRec1, xin0) \
                   + (1e-1) * torch.sum(torch.pow(self.embedding_model.koopmanOperator, 2))

            loss_reconstruct = loss_reconstruct + mseLoss(xRec1, xin0).detach()
            g1_old = g1Pred

        return loss, loss_reconstruct

    def evaluate(self, states):
        """Evaluates the embedding models reconstruction error and returns its
        predictions.
        Args:
            states (Tensor): [B, T, 3] Time-series feature tensor
        Returns:
            Tuple[Float, Tensor, Tensor]: Test error, Predicted states, Target states
        """
        self.embedding_model.eval()
        device = self.device

        mseLoss = nn.MSELoss()

        # Pull out targets from prediction dataset
        yTarget = states[:, 1:].to(device)
        xInput = states[:, :-1].to(device)
        yPred = torch.zeros(yTarget.size()).to(device)

        # Test accuracy of one time-step
        for i in range(xInput.size(1)):
            xInput0 = xInput[:, i].to(device)
            g0 = self.embedding_model.embed(xInput0)
            yPred0 = self.embedding_model.recover(g0)
            yPred[:, i] = yPred0.squeeze().detach()

        test_loss = mseLoss(yTarget, yPred)

        return test_loss, yPred, yTarget

    def predict_ahead(self,states,T):
        """Trains model for a single epoch
                Args:
                    states (Tensor): [B, T, 3] Time-series feature tensor
                Returns:
                    FloatTuple: Tuple containing:

                        | (float): Koopman based loss of current epoch
                        | (float): Reconstruction loss
                """
        self.embedding_model.eval()


        xin0 = states[:, 0].to(self.device)  # Time-step

        # Model forward for initial time-step
        g0, xRec0 = self.embedding_model(xin0)

        g1_old = g0
        predictions = []
        # Loop through time-series
        for t0 in range(0,T):
            g1Pred = self.embedding_model.koopmanOperation(g1_old)
            xgRec1 = self.embedding_model.recover(g1Pred)
            predictions.append(xgRec1.unsqueeze(1))
            g1_old = g1Pred

        return torch.concat(predictions,1)