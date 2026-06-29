"""
Agente SAC (Soft Actor-Critic) completo para TrackMania.

Integra ator, crítico, buffer de replay e lógica de treinamento
com gerenciamento adequado de gradientes, atualizações suaves
de redes-alvo e ajuste automático de temperatura de entropia.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from .models import AtorSAC, CriticoSAC
from .buffer import ReplayBuffer, RunningNormalizer


class AgenteSAC:
    """Agente Soft Actor-Critic completo para TrackMania."""

    def __init__(
        self,
        state_dim=83,
        action_dim=3,
        hidden_dim=256,
        lr_ator=3e-4,
        lr_critico=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_size=200_000,
        batch_size=256,
        target_entropy=None,
        device='cpu',
    ):
        """
        Inicializa o agente SAC.

        Args:
            state_dim: Dimensão do espaço de estados.
            action_dim: Dimensão do espaço de ações.
            hidden_dim: Dimensão das camadas ocultas das redes neurais.
            lr_ator: Taxa de aprendizado do ator.
            lr_critico: Taxa de aprendizado do crítico.
            lr_alpha: Taxa de aprendizado do coeficiente de entropia.
            gamma: Fator de desconto.
            tau: Coeficiente de atualização suave das redes-alvo.
            buffer_size: Tamanho máximo do buffer de replay.
            batch_size: Tamanho do mini-batch para treinamento.
            target_entropy: Entropia-alvo (padrão: -action_dim).
            device: Dispositivo de computação ('cpu' ou 'cuda').
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.target_entropy = target_entropy if target_entropy is not None else -action_dim

        # --- Redes neurais ---
        self.ator = AtorSAC(state_dim, action_dim, hidden_dim).to(self.device)
        self.critico = CriticoSAC(state_dim, action_dim, hidden_dim).to(self.device)

        # Rede-alvo do crítico (cópia profunda, sem gradientes)
        self.critico_alvo = copy.deepcopy(self.critico).to(self.device)
        for param in self.critico_alvo.parameters():
            param.requires_grad = False

        # --- Otimizadores ---
        self.otimizador_ator = torch.optim.Adam(self.ator.parameters(), lr=lr_ator)
        self.otimizador_critico = torch.optim.Adam(self.critico.parameters(), lr=lr_critico)

        # --- Coeficiente de entropia (alpha) aprendível ---
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.otimizador_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        self.alpha = self.log_alpha.exp().item()

        # --- Buffer de replay ---
        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

        # --- Normalizador de observações ---
        self.normalizador = RunningNormalizer(state_dim)

    def select_action(self, state, evaluate=False):
        """
        Seleciona uma ação dado o estado atual.

        Atualiza as estatísticas do normalizador e normaliza o estado
        antes de passá-lo pela rede do ator.

        Args:
            state: Vetor de estado (numpy array).
            evaluate: Se True, usa a ação determinística (média).

        Returns:
            Ação como numpy array.
        """
        # Atualizar estatísticas e normalizar
        self.normalizador.update(state)
        state_norm = self.normalizador.normalize(state)

        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                # Ação determinística (usa a média da distribuição)
                action, _, _ = self.ator.sample(state_tensor)
                # Para avaliação, podemos pegar a média diretamente
                mu = self.ator(state_tensor)
                action = torch.tanh(mu)
            else:
                action, _, _ = self.ator.sample(state_tensor)

        return action.cpu().numpy().flatten()

    def train_step(self):
        """
        Executa um passo de treinamento completo do SAC.

        Atualiza o crítico, o ator e o coeficiente alpha, seguido
        de uma atualização suave das redes-alvo.

        Returns:
            Dicionário com as perdas e métricas do passo:
            - loss_critico: Perda do crítico.
            - loss_ator: Perda do ator.
            - loss_alpha: Perda do alpha.
            - alpha: Valor atual do coeficiente de entropia.
            - q_medio: Valor Q médio estimado.
        """
        if len(self.buffer) < self.batch_size:
            return None

        # Amostrar batch do buffer de replay
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Converter para tensores no dispositivo correto
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # --- Atualização do Crítico ---
        with torch.no_grad():
            # Amostrar próxima ação da política atual
            next_actions, next_log_probs, _ = self.ator.sample(next_states)

            # Calcular Q-values alvo
            q1_target, q2_target = self.critico_alvo(next_states, next_actions)
            q_target_min = torch.min(q1_target, q2_target)

            # Alvo de Bellman com regularização de entropia
            target = rewards + self.gamma * (1.0 - dones) * (q_target_min - self.alpha * next_log_probs)

        # Q-values atuais
        q1, q2 = self.critico(states, actions)
        q_medio = torch.min(q1, q2).mean().item()

        # Perda do crítico (MSE)
        loss_critico = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.otimizador_critico.zero_grad()
        loss_critico.backward()
        torch.nn.utils.clip_grad_norm_(self.critico.parameters(), max_norm=1.0)
        self.otimizador_critico.step()

        # --- Atualização do Ator ---
        new_actions, log_probs, _ = self.ator.sample(states)
        q1_new, q2_new = self.critico(states, new_actions)
        q_min_new = torch.min(q1_new, q2_new)

        # Perda do ator: maximizar Q - alpha * entropia
        loss_ator = (self.alpha * log_probs - q_min_new).mean()

        self.otimizador_ator.zero_grad()
        loss_ator.backward()
        torch.nn.utils.clip_grad_norm_(self.ator.parameters(), max_norm=1.0)
        self.otimizador_ator.step()

        # --- Atualização do Alpha (temperatura de entropia) ---
        loss_alpha = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.otimizador_alpha.zero_grad()
        loss_alpha.backward()
        self.otimizador_alpha.step()

        # Atualizar alpha
        self.alpha = self.log_alpha.exp().item()

        # --- Atualização suave das redes-alvo ---
        with torch.no_grad():
            for param, target_param in zip(
                self.critico.parameters(), self.critico_alvo.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

        return {
            'loss_critico': loss_critico.item(),
            'loss_ator': loss_ator.item(),
            'loss_alpha': loss_alpha.item(),
            'alpha': self.alpha,
            'q_medio': q_medio,
        }

    def add_experience(self, state, action, reward, next_state, done):
        """
        Adiciona uma transição ao buffer de replay.

        Args:
            state: Estado atual.
            action: Ação tomada.
            reward: Recompensa recebida.
            next_state: Próximo estado.
            done: Se o episódio terminou (float: 0.0 ou 1.0).
        """
        self.buffer.add(state, action, reward, next_state, done)

    def save(self, path, episode):
        """
        Salva o checkpoint completo do agente.

        Inclui todos os pesos das redes, otimizadores, log_alpha
        e o normalizador de observações.

        Args:
            path: Caminho do arquivo para salvar.
            episode: Número do episódio atual (para referência).
        """
        checkpoint = {
            'episode': episode,
            'ator_state_dict': self.ator.state_dict(),
            'critico_state_dict': self.critico.state_dict(),
            'critico_alvo_state_dict': self.critico_alvo.state_dict(),
            'otimizador_ator_state_dict': self.otimizador_ator.state_dict(),
            'otimizador_critico_state_dict': self.otimizador_critico.state_dict(),
            'otimizador_alpha_state_dict': self.otimizador_alpha.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'normalizador': self.normalizador.get_state(),
        }
        torch.save(checkpoint, path)

    def load(self, path, episode=None):
        """
        Carrega o checkpoint completo do agente.

        Restaura todos os pesos das redes, otimizadores, log_alpha
        e o normalizador de observações.

        Args:
            path: Caminho do arquivo de checkpoint.
            episode: Não utilizado diretamente (mantido por compatibilidade).

        Returns:
            Número do episódio salvo no checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.ator.load_state_dict(checkpoint['ator_state_dict'])
        self.critico.load_state_dict(checkpoint['critico_state_dict'])
        self.critico_alvo.load_state_dict(checkpoint['critico_alvo_state_dict'])
        self.otimizador_ator.load_state_dict(checkpoint['otimizador_ator_state_dict'])
        self.otimizador_critico.load_state_dict(checkpoint['otimizador_critico_state_dict'])
        self.otimizador_alpha.load_state_dict(checkpoint['otimizador_alpha_state_dict'])

        self.log_alpha = checkpoint['log_alpha'].to(self.device).requires_grad_(True)
        # Reconstruir o otimizador de alpha com o novo parâmetro
        self.otimizador_alpha = torch.optim.Adam([self.log_alpha], lr=self.otimizador_alpha.defaults['lr'])
        self.alpha = self.log_alpha.exp().item()

        self.normalizador.set_state(checkpoint['normalizador'])

        return checkpoint.get('episode', 0)
