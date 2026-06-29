"""
Replay Buffer e normalizador online para o treinamento SAC.

Implementa armazenamento eficiente de transições usando arrays NumPy
pré-alocados e normalização de estados via algoritmo de Welford.
"""

import numpy as np
import torch
from typing import Dict, Optional


class ReplayBuffer:
    """
    Buffer de replay circular com armazenamento eficiente em arrays NumPy.

    Pré-aloca memória contígua para evitar overhead de alocação dinâmica
    e fragmentação de memória durante o treinamento.

    Args:
        state_dim: Dimensão do espaço de estados.
        action_dim: Dimensão do espaço de ações.
        max_size: Capacidade máxima do buffer.
    """

    def __init__(self, state_dim: int, action_dim: int, max_size: int = 200_000) -> None:
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Pré-alocação de arrays contíguos para armazenamento eficiente
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        """
        Adiciona uma transição ao buffer de forma circular.

        Quando o buffer está cheio, as transições mais antigas são
        sobrescritas automaticamente.

        Args:
            state: Estado atual.
            action: Ação tomada.
            reward: Recompensa recebida.
            next_state: Próximo estado.
            done: Se o episódio terminou.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Amostra um mini-batch aleatório de transições.

        Args:
            batch_size: Número de transições a amostrar.

        Returns:
            Dicionário com tensores float32:
                - 'estados': (batch, state_dim)
                - 'acoes': (batch, action_dim)
                - 'recompensas': (batch, 1)
                - 'prox_estados': (batch, state_dim)
                - 'dones': (batch, 1)
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            'estados': torch.as_tensor(self.states[indices], dtype=torch.float32),
            'acoes': torch.as_tensor(self.actions[indices], dtype=torch.float32),
            'recompensas': torch.as_tensor(self.rewards[indices], dtype=torch.float32),
            'prox_estados': torch.as_tensor(self.next_states[indices], dtype=torch.float32),
            'dones': torch.as_tensor(self.dones[indices], dtype=torch.float32),
        }

    def __len__(self) -> int:
        """Retorna o número atual de transições armazenadas."""
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """
        Verifica se o buffer tem transições suficientes para amostragem.

        Args:
            batch_size: Tamanho do mini-batch desejado.

        Returns:
            True se len(buffer) >= batch_size.
        """
        return self.size >= batch_size


class RunningNormalizer:
    """
    Normalizador online usando o algoritmo de Welford para cálculo
    incremental de média e variância.

    Permite normalizar observações em tempo real sem precisar armazenar
    todo o histórico de dados, ideal para normalização de estados em RL.

    Args:
        shape: Formato das observações (ex: (state_dim,)).
    """

    def __init__(self, shape: tuple) -> None:
        self.shape = shape
        self.count: int = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)

    @property
    def var(self) -> np.ndarray:
        """Variância amostral atual. Retorna 1.0 se count < 2."""
        if self.count < 2:
            return np.ones(self.shape, dtype=np.float64)
        return self.M2 / (self.count - 1)

    def update(self, x: np.ndarray) -> None:
        """
        Atualiza as estatísticas com nova(s) observação(ões).

        Suporta tanto observações individuais quanto batches.
        Para batches, cada linha é tratada como uma observação independente.

        Args:
            x: Observação com shape (shape,) ou batch com shape (N, *shape).
        """
        x = np.asarray(x, dtype=np.float64)

        if x.ndim == len(self.shape):
            # Observação individual
            self._update_single(x)
        else:
            # Batch de observações
            for obs in x:
                self._update_single(obs)

    def _update_single(self, x: np.ndarray) -> None:
        """
        Atualiza estatísticas com uma única observação (algoritmo de Welford).

        Args:
            x: Observação individual com shape igual a self.shape.
        """
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normaliza a observação usando média e variância correntes.

        Aplica z-score: (x - mean) / (sqrt(var) + eps)

        Args:
            x: Observação ou batch a ser normalizado.

        Returns:
            Observação normalizada como array NumPy float32.
        """
        x = np.asarray(x, dtype=np.float64)
        normalized = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return normalized.astype(np.float32)

    def save(self, path: str) -> None:
        """
        Salva as estatísticas em arquivo .npz.

        Args:
            path: Caminho do arquivo (extensão .npz adicionada automaticamente).
        """
        np.savez(
            path,
            mean=self.mean,
            M2=self.M2,
            count=np.array(self.count, dtype=np.int64),
        )

    def load(self, path: str) -> None:
        """
        Carrega as estatísticas de um arquivo .npz.

        Args:
            path: Caminho do arquivo .npz a carregar.
        """
        if not path.endswith('.npz'):
            path = path + '.npz'

        data = np.load(path)
        self.mean = data['mean'].astype(np.float64)
        self.M2 = data['M2'].astype(np.float64)
        self.count = int(data['count'])
