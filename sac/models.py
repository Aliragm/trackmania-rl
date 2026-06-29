"""
Redes neurais para o algoritmo Soft Actor-Critic (SAC).

Implementa o ator (política estocástica com squashing via tanh)
e o crítico (Twin Q-Networks) para aprendizado por reforço contínuo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class AtorSAC(nn.Module):
    """
    Rede do ator para SAC com política estocástica Gaussiana.

    Produz uma distribuição sobre ações contínuas usando o truque de
    reparametrização e squashing via tanh para limitar as ações a [-1, 1].

    Args:
        state_dim: Dimensão do espaço de estados.
        action_dim: Dimensão do espaço de ações.
        hidden_dim: Número de neurônios nas camadas ocultas.
    """

    def __init__(self, state_dim: int = 83, action_dim: int = 3, hidden_dim: int = 256) -> None:
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passa o estado pela rede e retorna média e log do desvio padrão.

        Args:
            x: Tensor de estados com shape (batch, state_dim) ou (state_dim,).

        Returns:
            Tupla (mu, log_std) onde ambos têm shape (..., action_dim).
        """
        h = self.shared(x)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Amostra uma ação usando o truque de reparametrização com squashing tanh.

        Aplica a correção de log-probabilidade para a transformação tanh:
            log_pi = log_prob_normal(x_t) - sum(log(1 - tanh(x_t)^2 + eps))

        Args:
            state: Tensor de estados com shape (batch, state_dim) ou (state_dim,).

        Returns:
            Tupla (action, log_prob) onde:
                - action: Ação squashed em [-1, 1] com shape (..., action_dim).
                - log_prob: Log-probabilidade escalar por amostra com shape (..., 1) ou (1,).
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mu, std)
        # Truque de reparametrização: x_t = mu + std * eps, eps ~ N(0, 1)
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Correção da log-probabilidade para a transformação tanh
        log_prob = normal.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Retorna a ação determinística (média) para avaliação.

        Args:
            state: Tensor de estados.

        Returns:
            Ação determinística tanh(mu) com shape (..., action_dim).
        """
        mu, _ = self.forward(state)
        return torch.tanh(mu)


class CriticoSAC(nn.Module):
    """
    Crítico Twin Q-Networks para SAC.

    Implementa duas redes Q independentes que recebem estado e ação
    concatenados como entrada. O valor mínimo entre as duas é usado
    para reduzir a superestimação (double Q-learning).

    Args:
        state_dim: Dimensão do espaço de estados.
        action_dim: Dimensão do espaço de ações.
        hidden_dim: Número de neurônios nas camadas ocultas.
    """

    def __init__(self, state_dim: int = 83, action_dim: int = 3, hidden_dim: int = 256) -> None:
        super().__init__()

        input_dim = state_dim + action_dim

        # Rede Q1
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Rede Q2
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula os valores Q para o par (estado, ação) usando ambas as redes.

        Args:
            state: Tensor de estados com shape (batch, state_dim).
            action: Tensor de ações com shape (batch, action_dim).

        Returns:
            Tupla (q1_value, q2_value) cada um com shape (batch, 1).
        """
        x = torch.cat([state, action], dim=-1)
        q1_value = self.q1(x)
        q2_value = self.q2(x)
        return q1_value, q2_value
