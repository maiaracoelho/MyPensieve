import multiprocessing as mp
import numpy as np
import pandas as pd

import logging
import os

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from utils import calculate_action_probabilities, sample_actions_from_probabilities

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()

from abr import ABREnv
import ppo2 as network

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

S_DIM = [8, 8]
A_DIM = 6
ACTOR_LR_RATE = 1e-4
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 1000
TRAIN_EPOCH = 10
MODEL_SAVE_INTERVAL = 10
RANDOM_SEED = 42
SUMMARY_DIR = "ppo"
TEST_LOG_FOLDER = "test_results/"
LOG_FILE = SUMMARY_DIR + "/log"
PPO_TRAINING_EPO = 5
ALGORITHM = "bb"  ## bb|stallion|lolypop
MODE = "qoeCost"  ## qoep|qoer|qoeCost
SCEN = "cloud"  ## edge|cloud|learn

# Criar diretórios necessários
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)
if not os.path.exists(TEST_LOG_FOLDER):
    os.makedirs(TEST_LOG_FOLDER)
if not os.path.exists(LOG_FILE):
    os.makedirs(LOG_FILE)

NN_MODEL = None


def testing(epoch, nn_model, log_file):
    """Função de teste para avaliar o modelo"""
    # Limpar resultados antigos
    for file in os.listdir(TEST_LOG_FOLDER):
        os.remove(os.path.join(TEST_LOG_FOLDER, file))

    # Executar script de teste
    os.system(f"python3 test.py {nn_model} {ALGORITHM} {MODE} {SCEN}")

    # Ler os logs com pandas
    rewards, entropies = [], []
    for test_log_file in os.listdir(TEST_LOG_FOLDER):
        log_path = os.path.join(TEST_LOG_FOLDER, test_log_file)

        # Leitura segura do arquivo usando pandas
        try:
            df = pd.read_csv(log_path, header=None)
            if df.shape[1] < 12:
                print(f"⚠️ Arquivo incompleto detectado: {test_log_file}")
                continue  # Ignorar arquivos incompletos

            # Extrair colunas corretas
            df.columns = [
                "time",
                "bitrate",
                "buffer",
                "rebuffering",
                "throughput",
                "delay",
                "throughput_kbps",
                "action",
                "qoe",
                "cost",
                "entropy",
                "reward",
            ]

            # Filtra apenas as colunas necessárias e remove linhas incompletas
            df.dropna(subset=["reward", "entropy"], inplace=True)

            rewards.append(df["reward"].mean())
            entropies.append(df["entropy"].mean())

        except Exception as e:
            print(f"⚠️ Erro ao ler o arquivo {test_log_file}: {e}")
            continue

    # Verificar se há dados válidos
    #if len(rewards) == 0 or len(entropies) == 0:
    #    print("⚠️ Nenhum dado de recompensa ou entropia encontrado!")

    # Estatísticas das recompensas
    rewards = np.array(rewards)
    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    # Escrevendo no arquivo de log
    log_file.write(
        f"{epoch}\t{rewards_min:.2f}\t{rewards_5per:.2f}\t{rewards_mean:.2f}"
        f"\t{rewards_median:.2f}\t{rewards_95per:.2f}\t{rewards_max:.2f}\n"
    )
    log_file.flush()
    print(
        f"✅ Escrevendo no log_test.txt: Época {epoch}, Recompensas {rewards_mean:.2f}"
    )

    return rewards_mean, np.mean(entropies)


def central_agent(net_params_queues, exp_queues):
    """Agente central que coordena os parâmetros e coleta experiências"""

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS
    tf_config = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    )
    with tf.Session(config=tf_config) as sess, open(
        LOG_FILE + "_" + ALGORITHM + "_test.txt", "w"
    ) as test_log_file:
        summary_ops, summary_vars = build_summaries()
        actor = network.Network(
            sess, state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE
        )
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        saver = tf.train.Saver(max_to_keep=1000)

        if NN_MODEL:
            saver.restore(sess, NN_MODEL)
            print("Modelo restaurado.")

        for epoch in range(TRAIN_EPOCH):
            # Sincronizar os parâmetros da rede com os agentes
            print(
            f"✅ Época {epoch}"
            )
            actor_net_params = actor.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put(actor_net_params)

            s, a, p, g = [], [], [], []
            for i in range(NUM_AGENTS):
                s_, a_, p_, g_ = exp_queues[i].get()
                s.extend(s_)
                a.extend(a_)
                p.extend(p_)
                g.extend(g_)

            s_batch = np.stack(s, axis=0)
            a_batch = np.vstack(a)
            p_batch = np.vstack(p)
            v_batch = np.vstack(g)

            for _ in range(PPO_TRAINING_EPO):
                actor.train(s_batch, a_batch, p_batch, v_batch, epoch)

            if epoch % MODEL_SAVE_INTERVAL == 0 or epoch == (TRAIN_EPOCH-1):
                save_path = saver.save(sess, f"{SUMMARY_DIR}/nn_model_ep_{epoch}.ckpt")
                avg_reward, avg_entropy = testing(epoch, save_path, test_log_file)
                summary_str = sess.run(
                    summary_ops,
                    feed_dict={
                        summary_vars[0]: actor._entropy_weight,
                        summary_vars[1]: avg_reward,
                        summary_vars[2]: avg_entropy,
                    },
                )
                writer.add_summary(summary_str, epoch)
                writer.flush()


def agent(agent_id, net_params_queue, exp_queue):
    """Agente individual que interage com o ambiente"""
    env = ABREnv(ALGORITHM, MODE, SCEN, agent_id)
    with tf.Session() as sess:
        actor = network.Network(
            sess, state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE
        )
        sess.run(tf.global_variables_initializer())

        for epoch in range(TRAIN_EPOCH):
            obs = env.reset()
            s_batch, a_batch, p_batch, r_batch = [], [], [], []

            for step in range(TRAIN_SEQ_LEN):
                s_batch.append(obs)

                action_prob = actor.predict(
                    np.reshape(obs, (1, S_DIM[0], S_DIM[1]))
                ).flatten()
                noisy_action_prob = calculate_action_probabilities(action_prob)
                decisions_eval = sample_actions_from_probabilities(noisy_action_prob)

                obs, rew, done, info, action_vec = env.step(decisions_eval)
                assert not np.any(np.isnan(obs)), "obs contém valores NaN"
                assert not np.any(np.isinf(obs)), "obs contém valores infinitos"

                a_batch.append(action_vec)
                r_batch.append(rew)
                p_batch.append(action_prob)

                if done:
                    break

            v_batch = actor.compute_v(s_batch, a_batch, r_batch, done)
            exp_queue.put([s_batch, a_batch, p_batch, v_batch])

            actor_net_params = net_params_queue.get()
            actor.set_network_params(actor_net_params)


def build_summaries():
    entropy_weight = tf.Variable(0.0, name="entropy_weight")
    avg_reward = tf.Variable(0.0, name="avg_reward")
    avg_entropy = tf.Variable(0.0, name="avg_entropy")

    tf.summary.scalar("Entropy_Weight", entropy_weight)
    tf.summary.scalar("Average_Reward", avg_reward)
    tf.summary.scalar("Average_Entropy", avg_entropy)

    summary_vars = [entropy_weight, avg_reward, avg_entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars



def main():
    """Função principal"""
    np.random.seed(RANDOM_SEED)

    net_params_queues = [mp.Queue(1) for _ in range(NUM_AGENTS)]
    exp_queues = [mp.Queue(1) for _ in range(NUM_AGENTS)]

    coordinator = mp.Process(target=central_agent, args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = [
        mp.Process(target=agent, args=(i, net_params_queues[i], exp_queues[i]))
        for i in range(NUM_AGENTS)
    ]
    for agent_proc in agents:
        agent_proc.start()

    coordinator.join()


if __name__ == "__main__":
    main()
