"""
TrackMania RL - Treinamento com SAC (Soft Actor-Critic)

Uso:
    python train_sac.py
    python train_sac.py --carregar pesos_sac/checkpoint_ep500.pth
    python train_sac.py --avaliar pesos_sac/checkpoint_ep500.pth
"""

import os
import time
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tmrl.envs import GenericGymEnv
from tmrl.config.config_objects import CONFIG_DICT

from sac.agent import AgenteSAC


# =============================================================================
# Funções auxiliares
# =============================================================================

def preparar_estado(obs):
    """
    Converte a observação do ambiente em um vetor de estado plano.

    A observação do TMRL é uma tupla com:
    - obs[0]: velocidade
    - obs[1]: dados LIDAR
    - obs[2:]: histórico de ações anteriores

    Returns:
        Vetor numpy 1D concatenando todos os componentes.
    """
    velocidade = np.array(obs[0]).flatten()
    lidar = np.array(obs[1]).flatten()
    acoes = [np.array(a).flatten() for a in obs[2:]]
    historico_acoes = np.concatenate(acoes) if len(acoes) > 0 else np.array([])
    return np.concatenate([velocidade, lidar, historico_acoes])


def calcular_recompensa(recompensa_crua, proximo_obs, done, passo, max_passos):
    """
    Aplica reward shaping à recompensa crua do ambiente.

    Adiciona bônus por velocidade, punições por lentidão, crash
    e proximidade excessiva com paredes (LIDAR).

    Args:
        recompensa_crua: Recompensa original do ambiente.
        proximo_obs: Próxima observação do ambiente.
        done: Se o episódio terminou.
        passo: Passo atual dentro do episódio.
        max_passos: Número máximo de passos por episódio.

    Returns:
        Recompensa modificada com reward shaping.
    """
    recompensa = recompensa_crua
    velocidade_kmh = proximo_obs[0][0]

    # Bônus por velocidade quando progredindo
    if recompensa_crua > 0:
        recompensa += velocidade_kmh * 0.02

    # Punição por lentidão
    if velocidade_kmh < 5.0:
        recompensa -= 0.5

    # Punição por crash (episódio terminou prematuramente)
    if done and passo < max_passos - 1:
        recompensa -= 10.0

    # Punição por raspar na parede (LIDAR)
    try:
        lidar_data = np.array(proximo_obs[1])
        if lidar_data.ndim > 1:
            lidar_recente = lidar_data[-1]
        else:
            lidar_recente = lidar_data
        menor_distancia = np.min(lidar_recente)
        if menor_distancia < 0.08 and not done:
            recompensa -= 0.5
    except Exception:
        pass

    return recompensa


def plotar_curvas(historico_recompensas, historico_loss_ator, historico_loss_critico,
                  historico_alpha, pasta_pesos):
    """
    Plota e salva as curvas de treinamento em 3 subplots.

    Args:
        historico_recompensas: Lista de recompensas por episódio.
        historico_loss_ator: Lista de perdas do ator.
        historico_loss_critico: Lista de perdas do crítico.
        historico_alpha: Lista de valores de alpha.
        pasta_pesos: Diretório onde salvar a imagem.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Recompensas
    axes[0].plot(historico_recompensas, alpha=0.3, color='blue', label='Recompensa')
    if len(historico_recompensas) >= 50:
        media_movel = np.convolve(historico_recompensas, np.ones(50) / 50, mode='valid')
        axes[0].plot(range(49, len(historico_recompensas)), media_movel,
                     color='red', linewidth=2, label='Média (50 ep)')
    axes[0].set_ylabel('Recompensa')
    axes[0].set_title('Treinamento SAC - TrackMania')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Perdas
    if historico_loss_ator:
        axes[1].plot(historico_loss_ator, alpha=0.5, label='Perda Ator', color='green')
    if historico_loss_critico:
        axes[1].plot(historico_loss_critico, alpha=0.5, label='Perda Crítico', color='orange')
    axes[1].set_ylabel('Perda')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Alpha
    if historico_alpha:
        axes[2].plot(historico_alpha, color='purple', label='Alpha')
    axes[2].set_xlabel('Episódio')
    axes[2].set_ylabel('Alpha')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    caminho_plot = os.path.join(pasta_pesos, 'curvas_treinamento.png')
    plt.savefig(caminho_plot, dpi=150)
    plt.close()
    print(f"\n[INFO] Curvas de treinamento salvas em: {caminho_plot}")


# =============================================================================
# Treinamento
# =============================================================================

def treinar(args):
    """
    Loop principal de treinamento do agente SAC no TrackMania.

    Gerencia warmup, coleta de experiência, atualização das redes,
    checkpoints periódicos e salvamento do melhor modelo.

    Args:
        args: Argumentos de linha de comando.
    """
    print("=" * 60)
    print("  TrackMania RL - Treinamento SAC")
    print("=" * 60)

    # Criar diretório de pesos
    os.makedirs(args.pasta_pesos, exist_ok=True)

    # Dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Dispositivo: {device}")

    # Criar ambiente
    print("[INFO] Criando ambiente TrackMania...")
    env = GenericGymEnv(id='real-time-gym-ts-v1', gym_kwargs={'config': CONFIG_DICT})

    # Detectar dimensão do estado a partir da primeira observação
    print("[INFO] Detectando dimensões do espaço de estados...")
    obs, info = env.reset()
    state = preparar_estado(obs)
    state_dim = len(state)
    action_dim = 3
    print(f"[INFO] state_dim={state_dim}, action_dim={action_dim}")

    # Criar agente
    agente = AgenteSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lr_ator=3e-4,
        lr_critico=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        device=device,
    )

    # Carregar checkpoint se especificado
    episodio_inicial = 0
    if args.carregar:
        if os.path.exists(args.carregar):
            episodio_inicial = agente.load(args.carregar)
            print(f"[INFO] Checkpoint carregado de: {args.carregar} (episódio {episodio_inicial})")
        else:
            print(f"[AVISO] Arquivo não encontrado: {args.carregar}. Iniciando do zero.")

    # Históricos de treinamento
    historico_recompensas = []
    historico_loss_ator = []
    historico_loss_critico = []
    historico_alpha = []
    melhor_recompensa = -float('inf')
    total_steps = 0

    print(f"\n[INFO] Iniciando treinamento: {args.max_episodios} episódios")
    print(f"[INFO] Warmup: {args.warmup} passos | Atualização a cada {args.update_every} passos")
    print(f"[INFO] Gradient steps por atualização: {args.gradient_steps}")
    print("-" * 60)

    try:
        for ep in range(episodio_inicial, args.max_episodios):
            obs, info = env.reset()
            recompensa_ep = 0.0
            train_info = None
            tempo_inicio = time.time()

            for passo in range(args.max_passos):
                state = preparar_estado(obs)

                # Warmup: ações aleatórias
                if total_steps < args.warmup:
                    action = np.random.uniform(-1, 1, size=action_dim)
                else:
                    action = agente.select_action(state, evaluate=False)

                # Executar ação no ambiente
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Reward shaping
                shaped_reward = calcular_recompensa(
                    reward, next_obs, done, passo, args.max_passos
                )

                next_state = preparar_estado(next_obs)

                # Armazenar experiência
                agente.add_experience(state, action, shaped_reward, next_state, float(done))

                # Treinamento
                if total_steps >= args.warmup and total_steps % args.update_every == 0:
                    for _ in range(args.gradient_steps):
                        train_info = agente.train_step()

                recompensa_ep += shaped_reward
                total_steps += 1

                if done:
                    break

                obs = next_obs

            # Registrar histórico
            historico_recompensas.append(recompensa_ep)

            loss_ator_ep = train_info['loss_ator'] if train_info else 0.0
            loss_critico_ep = train_info['loss_critico'] if train_info else 0.0
            alpha_ep = train_info['alpha'] if train_info else agente.alpha
            q_medio_ep = train_info['q_medio'] if train_info else 0.0

            historico_loss_ator.append(loss_ator_ep)
            historico_loss_critico.append(loss_critico_ep)
            historico_alpha.append(alpha_ep)

            tempo_ep = time.time() - tempo_inicio

            # Atualizar melhor recompensa
            if recompensa_ep > melhor_recompensa:
                melhor_recompensa = recompensa_ep
                caminho_melhor = os.path.join(args.pasta_pesos, 'melhor_modelo.pth')
                agente.save(caminho_melhor, ep + 1)

            # Log
            warmup_str = " [WARMUP]" if total_steps < args.warmup else ""
            print(
                f"Ep {ep + 1}/{args.max_episodios} | "
                f"Passos: {passo + 1} | "
                f"Recomp: {recompensa_ep:.1f} | "
                f"Q medio: {q_medio_ep:.2f} | "
                f"alpha: {alpha_ep:.4f} | "
                f"Loss A: {loss_ator_ep:.4f} | "
                f"Loss C: {loss_critico_ep:.4f} | "
                f"Melhor: {melhor_recompensa:.1f} | "
                f"Tempo: {tempo_ep:.1f}s"
                f"{warmup_str}"
            )

            # Checkpoint periódico
            if (ep + 1) % args.save_every == 0:
                caminho_checkpoint = os.path.join(
                    args.pasta_pesos, f'checkpoint_ep{ep + 1}.pth'
                )
                agente.save(caminho_checkpoint, ep + 1)
                print(f"  >> Checkpoint salvo: {caminho_checkpoint}")

    except KeyboardInterrupt:
        print("\n[INFO] Treinamento interrompido pelo usuário.")

    except Exception as e:
        print(f"\n[ERRO] Exceção durante treinamento: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Salvar pesos finais
        caminho_final = os.path.join(args.pasta_pesos, 'modelo_final.pth')
        agente.save(caminho_final, len(historico_recompensas))
        print(f"\n[INFO] Pesos finais salvos em: {caminho_final}")

        # Plotar curvas de treinamento
        if historico_recompensas:
            plotar_curvas(
                historico_recompensas,
                historico_loss_ator,
                historico_loss_critico,
                historico_alpha,
                args.pasta_pesos,
            )

        print(f"[INFO] Total de passos: {total_steps}")
        print(f"[INFO] Melhor recompensa: {melhor_recompensa:.1f}")
        print("[INFO] Treinamento finalizado.")


# =============================================================================
# Avaliação
# =============================================================================

def avaliar(args):
    """
    Avalia um agente SAC treinado no ambiente TrackMania.

    Executa episódios sem treinamento, usando ações determinísticas,
    e imprime os resultados.

    Args:
        args: Argumentos de linha de comando.
    """
    print("=" * 60)
    print("  TrackMania RL - Avaliação SAC")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Dispositivo: {device}")

    # Criar ambiente
    print("[INFO] Criando ambiente TrackMania...")
    env = GenericGymEnv(id='real-time-gym-ts-v1', gym_kwargs={'config': CONFIG_DICT})

    # Detectar dimensão do estado
    obs, info = env.reset()
    state = preparar_estado(obs)
    state_dim = len(state)
    action_dim = 3
    print(f"[INFO] state_dim={state_dim}, action_dim={action_dim}")

    # Criar e carregar agente
    agente = AgenteSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )

    if not os.path.exists(args.avaliar):
        print(f"[ERRO] Arquivo de checkpoint não encontrado: {args.avaliar}")
        return

    episodio_salvo = agente.load(args.avaliar)
    print(f"[INFO] Modelo carregado de: {args.avaliar} (episódio {episodio_salvo})")

    recompensas = []
    num_episodios_avaliacao = 10

    print(f"\n[INFO] Executando {num_episodios_avaliacao} episódios de avaliação...")
    print("-" * 60)

    for ep in range(num_episodios_avaliacao):
        obs, info = env.reset()
        recompensa_ep = 0.0
        tempo_inicio = time.time()

        for passo in range(args.max_passos):
            state = preparar_estado(obs)
            action = agente.select_action(state, evaluate=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            shaped_reward = calcular_recompensa(
                reward, next_obs, done, passo, args.max_passos
            )
            recompensa_ep += shaped_reward

            if done:
                break

            obs = next_obs

        tempo_ep = time.time() - tempo_inicio
        recompensas.append(recompensa_ep)

        print(
            f"Ep Avaliação {ep + 1}/{num_episodios_avaliacao} | "
            f"Passos: {passo + 1} | "
            f"Recompensa: {recompensa_ep:.1f} | "
            f"Tempo: {tempo_ep:.1f}s"
        )

    print("-" * 60)
    print(f"Recompensa média: {np.mean(recompensas):.1f} ± {np.std(recompensas):.1f}")
    print(f"Recompensa mínima: {np.min(recompensas):.1f}")
    print(f"Recompensa máxima: {np.max(recompensas):.1f}")
    print("[INFO] Avaliação finalizada.")


# =============================================================================
# Argumentos de linha de comando
# =============================================================================

def parse_args():
    """Processa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description='TrackMania RL - Treinamento com SAC',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--max-episodios', type=int, default=5000,
                        help='Número máximo de episódios de treinamento')
    parser.add_argument('--max-passos', type=int, default=5000,
                        help='Número máximo de passos por episódio')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Tamanho do mini-batch')
    parser.add_argument('--buffer-size', type=int, default=200000,
                        help='Tamanho do buffer de replay')
    parser.add_argument('--warmup', type=int, default=1000,
                        help='Passos com ações aleatórias antes de iniciar o treinamento')
    parser.add_argument('--update-every', type=int, default=2,
                        help='Frequência de atualização (a cada N passos)')
    parser.add_argument('--gradient-steps', type=int, default=1,
                        help='Número de passos de gradiente por atualização')
    parser.add_argument('--save-every', type=int, default=100,
                        help='Salvar checkpoint a cada N episódios')
    parser.add_argument('--carregar', type=str, default=None,
                        help='Caminho para carregar checkpoint e continuar treinamento')
    parser.add_argument('--avaliar', type=str, default=None,
                        help='Caminho para carregar modelo e avaliar (sem treinamento)')
    parser.add_argument('--pasta-pesos', type=str, default='pesos_sac',
                        help='Diretório para salvar pesos e checkpoints')

    return parser.parse_args()


# =============================================================================
# Ponto de entrada
# =============================================================================

if __name__ == '__main__':
    args = parse_args()

    if args.avaliar:
        avaliar(args)
    else:
        treinar(args)
