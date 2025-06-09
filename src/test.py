import os
import sys
import multiprocessing as mp

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Força CPU no processo de teste

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from utils import calculate_action_probabilities, sample_actions_from_probabilities

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()

import load_trace

# import a2c as network
import ppo2 as network
import RewardMetrics as r
from bb import bb_algo
from stallion import Stallion
from lolypop import Lolypop

import fixed_env as env

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


S_INFO = 8  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end, action_vec
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = np.array([300.0, 750.0, 1200.0, 1850.0, 2850.0, 4300.0])  # Kbps
BUFFER_NORM_FACTOR = 12.0
BMIN = 4.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
LOG_FILE = "test_results/log_sim_ppo"
TEST_TRACES = "./test/"
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = sys.argv[1]
algorithm = sys.argv[2]
mode = sys.argv[3]
scen = sys.argv[4]


def run_algorithm(algorithm, traces=TEST_TRACES):

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(traces)

    net_env = env.Environment(
        all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw
    )

    MAX_VIDEOS = 10  # Limitar para execução rápida

    log_path = LOG_FILE + "_" + algorithm + "_" + scen + "_" + all_file_names[net_env.trace_idx]
    log_file = open(log_path, "w")

    with tf.Session() as sess:
        actor = network.Network(
            sess,
            state_dim=[S_INFO, S_LEN],
            action_dim=A_DIM,
            learning_rate=ACTOR_LR_RATE,
        )

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        for var in tf.global_variables():
            weight = sess.run(var)

            assert not np.any(np.isnan(weight)), f"Peso contém NaN: {var.name}"
            assert not np.any(np.isinf(weight)), f"Peso contém infinito: {var.name}"

        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")

        time_stamp_ms = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = last_bit_rate
        recommended_rates = np.ones(A_DIM)
        state = np.zeros((S_INFO, S_LEN))
        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = []
        r_batch = []
        p_batch = []
        entropy_record = []
        entropy_ = 0.5
        video_count = 0

        rew = r.RewardMetrics(BMIN, min(VIDEO_BIT_RATE), max(VIDEO_BIT_RATE))

        if algorithm == "stallion":
            algo_instance = Stallion(
                video_bit_rate=VIDEO_BIT_RATE,
                window_size=8,
                z_thr=0.3,
                z_latency=0.75,
                lat_threshold=1.5,
                min_stable_steps=2,
            )
        elif algorithm == "lolypop":
            SIGMA_STAR = 0.3  # Limite de segmentos ignorados
            OMEGA_STAR = 1.0  # Limite de transições de qualidade

            lolypop = Lolypop(SIGMA_STAR, OMEGA_STAR, DEFAULT_QUALITY)

            current_transitions = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            action = recommended_rates * VIDEO_BIT_RATE
            a_batch.append(recommended_rates)


            if VIDEO_BIT_RATE[bit_rate] not in action:
                    delay_factor = 10.0
            else:
                    delay_factor = 1.0

            (
                delay_ms,
                sleep_ms,
                buffer_size_s,
                rebuf_s,
                video_chunk_size,
                next_video_chunk_sizes,
                end_of_video,
                video_chunk_remain,
                raw_throughput_bytes_s,
            ) = net_env.get_video_chunk(bit_rate, delay_factor)

            # Atualiza tempo total em ms
            time_stamp_ms += delay_ms
            time_stamp_ms += sleep_ms

            # Converte throughput para kbps
            throughput_kbps = (raw_throughput_bytes_s * 8) / 1000.0
            # Converte time_stamp para segundos ao salvar no log
            time_s = time_stamp_ms / 1000.0

            # reward is video quality - rebuffer penalty - smoothness

            bitrate_arrays=[]
            if scen=="cloud":
                bitrate_arrays = VIDEO_BIT_RATE
            else:
                bitrate_arrays = action
            data = {
                "bit_rate": VIDEO_BIT_RATE[bit_rate],
                "rebufering_time": rebuf_s,
                "last_bit_rate": VIDEO_BIT_RATE[last_bit_rate],
                "max_bit_rate": np.max(bitrate_arrays),
                "buffer_size": buffer_size_s,
                "delay": delay_ms,
                "video_chunk_size": video_chunk_size,
                "next_video_chunk_sizes": next_video_chunk_sizes,
                "action": bitrate_arrays,
                "throughput": throughput_kbps,
            }

            reward = rew.calculate_reward(data, mode, scen)
            qoe = rew.calculate_reward(data, "qoer", scen)
            cost = rew.calculate_reward(data, "cost", scen)
            r_batch.append(reward)

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(
                "\n"
                + str(time_s)
                + ","
                + str(VIDEO_BIT_RATE[bit_rate])
                + ","
                + str(buffer_size_s)
                + ","
                + str(rebuf_s)
                + ","
                + str(video_chunk_size)
                + ","
                + str(delay_ms / M_IN_K)
                + ","
                + str(throughput_kbps)
                + ","
                + str(action)
                + ","
                + str(qoe)
                + ","
                + str(cost)
                + ","
                + str(entropy_)
                + ","
                + str(reward)
            )
            log_file.flush()

            # retrieve previous state
            if len(s_batch) == 0:
                state0 = np.zeros((S_INFO, S_LEN))
            else:
                state0 = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state0, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(
                np.max(VIDEO_BIT_RATE)
            )  # last quality
            state[1, -1] = buffer_size_s / BUFFER_NORM_FACTOR
            state[2, -1] = (
                float(video_chunk_size) / float(delay_ms) / M_IN_K
            )  # kilo byte / ms
            state[3, -1] = float(delay_ms) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = (
                np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
            )  # mega byte
            state[5, -1] = np.minimum(
                video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP
            ) / float(CHUNK_TIL_VIDEO_END_CAP)
            state[6, -1] = rebuf_s / BUFFER_NORM_FACTOR  # 10 sec
            state[7, :A_DIM] = action / np.max(VIDEO_BIT_RATE)
            assert not np.any(np.isnan(state)), "Inputs têm valores NaN"
            assert not np.any(np.isinf(state)), "Inputs têm valores infinitos"

            s_batch.append(state)

            last_bit_rate = bit_rate

            if algorithm == "bb":
                # BB: decide baseado no buffer
                bit_rate = bb_algo(
                    buffer_size_s, VIDEO_BIT_RATE, DEFAULT_QUALITY
                )
            elif algorithm == "stallion":
                latency_s = delay_ms / M_IN_K
                algo_instance.update_metrics(throughput_kbps, latency_s)
                bit_rate = algo_instance.select_quality()
            elif algorithm == "lolypop":
                probabilities = [
                (
                    min(
                        1.0,
                        buffer_size_s / ((next_video_chunk_sizes[j] * 8) / (throughput_kbps * 1000)),
                    )
                    if throughput_kbps > 0
                    else 0.0
                )
                for j in range(len(VIDEO_BIT_RATE))
                ]

                # Atualizar transições de qualidade se necessário
                if video_chunk_remain > 0:
                    current_transitions += 1 / video_chunk_remain

                bit_rate = lolypop.select_representation(
                    probabilities, current_transitions
                )


            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN))).flatten()
            p_batch.append(action_prob)

            noisy_action_prob = calculate_action_probabilities(action_prob)
            decisions_eval = sample_actions_from_probabilities(noisy_action_prob)
            # print("decisions_eval no test", decisions_eval)

            if scen=="cloud":
                recommended_rates = np.zeros(A_DIM)
            elif scen=="edge":
                recommended_rates = np.ones(A_DIM)
            else:
                recommended_rates = decisions_eval

            entropy_ = -np.dot(action_prob, np.log(action_prob + 1e-8))
            entropy_record.append(entropy_)

            if end_of_video:
                log_file.close()

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del p_batch[:]

                action = recommended_rates * VIDEO_BIT_RATE

                if algorithm == "lolypop":
                    current_transitions = 0

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(recommended_rates)
                # print(np.mean(entropy_record))
                entropy_record = []

                video_count += 1

                if video_count >= min(len(all_file_names), MAX_VIDEOS):
                #if video_count >= len(all_file_names):

                    print(f"✅ Vídeo {video_count}/{len(all_file_names)} concluído")

                    break

                log_path = (
                    LOG_FILE + "_" + algorithm + "_" + scen +  "_" + all_file_names[net_env.trace_idx]
                )
                log_file = open(log_path, "w")


def main():
    np.random.seed(RANDOM_SEED)

    run_algorithm(algorithm, TEST_TRACES)
    print("Execução concluída. Logs salvos")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()

