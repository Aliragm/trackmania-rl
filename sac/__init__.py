"""
Módulo SAC (Soft Actor-Critic) para TrackMania RL.

Exporta as redes neurais (ator e crítico) e utilitários de buffer/normalização.
"""

from .models import AtorSAC, CriticoSAC
from .buffer import ReplayBuffer, RunningNormalizer

__all__ = [
    "AtorSAC",
    "CriticoSAC",
    "ReplayBuffer",
    "RunningNormalizer",
]
