from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.autograd import Variable
from ttvae.util import DataTransformer, reparameterize, _loss_function_MMD,z_gen
# from ttvae.base import BaseSynthesizer, random_state


class Encoder_T(nn.Module):
    def __init__(self, input_dim, latent_dim, embedding_dim, nhead, dim_feedforward=2048, dropout=0.1):
      super(Encoder_T, self).__init__()
      # Input data to Transformer
      self.linear = nn.Linear(input_dim,embedding_dim)
      # Transformer Encoder
      self.transformerencoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead, dim_feedforward, dropout)
      self.encoder = nn.TransformerEncoder(self.transformerencoder_layer, num_layers=2)
      # Latent Space Representation
      self.fc_mu = nn.Linear(embedding_dim, latent_dim)
      self.fc_log_var = nn.Linear(embedding_dim, latent_dim)

    def forward(self, x):
      # Encoder
      x = self.linear(x)
      enc_output = self.encoder(x)
      # Latent Space Representation
      mu = self.fc_mu(enc_output)
      logvar = self.fc_log_var(enc_output)
      std = torch.exp(0.5 * logvar)
      return mu, std, logvar, enc_output


class Decoder_T(nn.Module):
    def __init__(self, input_dim, latent_dim, embedding_dim, nhead, dim_feedforward=2048, dropout=0.1):
      super(Decoder_T, self).__init__()
      # Linear layer for mapping latent space to decoder input size
      self.latent_to_decoder_input = nn.Linear(latent_dim, embedding_dim)
      # Transformer Decoder
      self.transformerdecoder_layer = nn.TransformerDecoderLayer(embedding_dim, nhead, dim_feedforward, dropout)
      self.decoder = nn.TransformerDecoder(self.transformerdecoder_layer, num_layers=2)
      # Transformer Embedding to input
      self.linear = nn.Linear(embedding_dim,input_dim)
      self.sigma = Parameter(torch.ones(input_dim) * 0.1)

    def forward(self, z, enc_output):
      # Encoder
      z_decoder_input = self.latent_to_decoder_input(z)
      # Decoder
      # Note: Pass enc_output (memory) to the decoder
      dec_output = self.decoder(z_decoder_input, enc_output)

      return self.linear(dec_output), self.sigma


class TTVAE():
    """TTVAE."""

    def __init__(
        self,
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        latent_dim =32,# Example latent dimension
        embedding_dim=128,# Transformer embedding dimension
        nhead=8,# Number of attention heads
        dim_feedforward=1028,# Feedforward layer dimension
        dropout=0.1,
        cuda=True,
        verbose=False,
        device='cuda'
    ):
        self.latent_dim=latent_dim
        self.embedding_dim = embedding_dim
        self.nhead=nhead
        self.dim_feedforward=dim_feedforward
        self.dropout=dropout
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self._device = torch.device(device)

    # @random_state
    def fit(self, train_data, discrete_columns=(),save_path=''):
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)

        self.train_data = self.transformer.transform(train_data).astype('float32')
        dataset = TensorDataset(torch.from_numpy(self.train_data).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions

        print(data_dim, self.latent_dim, self.embedding_dim, self.nhead, self.dim_feedforward, self.dropout)

        self.encoder = Encoder_T(data_dim, self.latent_dim, self.embedding_dim, self.nhead, self.dim_feedforward, self.dropout).to(self._device)
        self.decoder = Decoder_T(data_dim, self.latent_dim, self.embedding_dim, self.nhead, self.dim_feedforward, self.dropout).to(self._device)

        optimizer = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

        self.encoder.train()
        self.decoder.train()

        best_loss = float('inf')
        patience = 0
        start_time = time.time()

        for epoch in range(self.epochs):
        
            pbar = tqdm(loader, total=len(loader))
            pbar.set_description(f"Epoch {epoch+1}/{self.epochs}")

            batch_loss = 0.0
            len_input = 0

            for id_, data in enumerate(pbar):
                optimizer.zero_grad()
                real_x = data[0].to(self._device)
                mean, std, logvar, enc_output = self.encoder(real_x)
                z = reparameterize(mean, logvar)
                recon_x, sigmas = self.decoder(z,enc_output)
                loss = _loss_function_MMD(recon_x, real_x, sigmas, mean, logvar, self.transformer.output_info_list, self.loss_factor)

                batch_loss += loss.item() * len(real_x)
                len_input += len(real_x)

                loss.backward()
                optimizer.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                pbar.set_postfix({"Loss": loss.item()})

            curr_loss = batch_loss/len_input
            scheduler.step(curr_loss)

            if curr_loss < best_loss:
              best_loss = loss.item()
              patience = 0
              torch.save(self, save_path+'/model.pt')
            else:
                patience += 1
                if patience == 500:
                    print('Early stopping')
                    break

                        
    # @random_state
    def sample(self, n_samples=100):
        """Sample data similar to the training data.

        """
        self.encoder.eval()
        with torch.no_grad():
            mean, std, logvar, enc_embed= self.encoder(torch.Tensor(self.train_data).to(self._device))

        embeddings = torch.normal(mean=mean, std=std).cpu().detach().numpy()
        synthetic_embeddings=z_gen(embeddings,n_to_sample=n_samples,metric='minkowski',interpolation_method='SMOTE')
        noise = torch.Tensor(synthetic_embeddings).to(self._device)

        self.decoder.eval()
        with torch.no_grad():
          fake, sigmas = self.decoder(noise,enc_embed)
          fake = torch.tanh(fake).cpu().detach().numpy()

        return self.transformer.inverse_transform(fake)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)