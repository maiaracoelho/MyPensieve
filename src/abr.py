# add queuing delay into halo
import os
import numpy as np
import core as abrenv
import load_trace
import RewardMetrics as r

from bb import bb_algo
from stallion import Stallion
from lolypop import Lolypop


# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 8
S_LEN = 8  # take how many frames in the past
A_DIM = 6
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = np.array([300.0, 750.0, 1200.0, 1850.0, 2850.0, 4300.0])  # Kbps
BUFFER_NORM_FACTOR = 12.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
EPS = 1e-6
BMIN = 4.0


class ABREnv:
    def __init__(self, algorithm, mode, scen, random_seed=RANDOM_SEED):
        np.random.seed(random_seed)
        all_cooked_time, all_cooked_bw, _ = load_trace.load_trace()
        self.net_env = abrenv.Environment(
            all_cooked_time=all_cooked_time,
            all_cooked_bw=all_cooked_bw,
            random_seed=random_seed,
        )

        self.last_bit_rate = DEFAULT_QUALITY
        self.bit_rate = self.last_bit_rate
        self.buffer_size = 0.0
        self.reward = r.RewardMetrics(BMIN, min(VIDEO_BIT_RATE), max(VIDEO_BIT_RATE))
        self.state = np.zeros((S_INFO, S_LEN))
        self.algorithm = algorithm
        self.mode = mode
        self.scen = scen
        # self.reset()

    def seed(self, num):
        np.random.seed(num)

    def reset(self):
        # self.net_env.reset_ptr()
        self.time_stamp = 0
        self.last_bit_rate = DEFAULT_QUALITY
        self.state = np.zeros((S_INFO, S_LEN))
        self.buffer_size = 0.0
        self.bit_rate = self.last_bit_rate

        if self.algorithm == "stallion":
            self.algo_instance = Stallion(
                video_bit_rate=VIDEO_BIT_RATE,
                window_size=8,
                z_thr=0.3,
                z_latency=0.75,
                lat_threshold=1.5,
                min_stable_steps=2,
            )

        elif self.algorithm == "lolypop":


            self.algo_instance = Lolypop(
                sigma_star=0.3, omega_star=1.0, DEFAULT_QUALITY=0
            )

            self.current_transitions = 0
        (
            delay,
            sleep_time,
            self.buffer_size,
            rebuf,
            video_chunk_size,
            next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain,
            throughput,
        ) = self.net_env.get_video_chunk(self.bit_rate, 1.0)

        state = np.roll(self.state, -1, axis=1)


        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[self.bit_rate] / float(
            np.max(VIDEO_BIT_RATE)
        )  # last quality
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = (
            np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
        )  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(
            CHUNK_TIL_VIDEO_END_CAP
        )
        state[6, -1] = 0 / BUFFER_NORM_FACTOR
        state[7, :A_DIM] = np.array(VIDEO_BIT_RATE) / M_IN_K / M_IN_K
        self.state = state

        return state
        # return state.reshape((1, S_INFO*S_LEN))

    def render(self):
        return

    def step(self, recommended_rates):

        #if self.scen == "edge":
        ##    recommended_rates = np.ones(A_DIM)  # borda tem tudo
        #elif self.scen == "cloud":
        #    recommended_rates = np.zeros(A_DIM)  # cloud não tem nada local
        #elif recommended_rates is None:
        #    raise ValueError("recommended_rates must be provided in 'learn' mode")

        action = recommended_rates * VIDEO_BIT_RATE

        delay_factor = 1.0
        if VIDEO_BIT_RATE[self.bit_rate] not in action:
            delay_factor = 10.0
        (
            delay,
            sleep_time,
            self.buffer_size,
            rebuf,
            video_chunk_size,
            next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain,
            throughput,
        ) = self.net_env.get_video_chunk(self.bit_rate, delay_factor)

        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        bitrate_arrays=[]
        if self.scen=="cloud":
                bitrate_arrays = VIDEO_BIT_RATE
        else:
                bitrate_arrays = action
        data = {
            "bit_rate": VIDEO_BIT_RATE[self.bit_rate],
            "rebufering_time": rebuf,
            "last_bit_rate": VIDEO_BIT_RATE[self.last_bit_rate],
            "max_bit_rate": np.max(bitrate_arrays),
            "buffer_size": self.buffer_size,
            "delay": delay,
            "video_chunk_size": video_chunk_size,
            "next_video_chunk_sizes": next_video_chunk_sizes,
            "action": bitrate_arrays,
            "throughput": throughput,
        }

        reward = self.reward.calculate_reward(data, self.mode, self.scen)

        state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[self.bit_rate] / float(
            np.max(VIDEO_BIT_RATE)
        )  # last quality
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = (
            np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
        )  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(
            CHUNK_TIL_VIDEO_END_CAP
        )
        state[6, -1] = rebuf / BUFFER_NORM_FACTOR  # 10 sec
        state[7, :A_DIM] = action / M_IN_K / M_IN_K

        assert not np.any(np.isnan(state)), "Inputs têm valores NaN"
        assert not np.any(np.isinf(state)), "Inputs têm valores infinitos"

        self.state = state

        self.last_bit_rate = self.bit_rate

        if self.algorithm == "bb":
            # BB: decide baseado no buffer
            self.bit_rate = bb_algo(
                self.buffer_size, VIDEO_BIT_RATE, DEFAULT_QUALITY
            )
        elif self.algorithm == "stallion":
            latency_s = delay / M_IN_K
            self.algo_instance.update_metrics(throughput, latency_s)
            self.bit_rate = self.algo_instance.select_quality()
        elif self.algorithm == "lolypop":


            probabilities = [
                (
                    min(
                        1.0,
                        self.buffer_size / ((next_video_chunk_sizes[j] * 8) / (throughput * 1000)),
                    )
                    if throughput > 0
                    else 0.0
                )
                for j in range(len(VIDEO_BIT_RATE))
                ]

            # Atualizar transições de qualidade se necessário
            if video_chunk_remain > 0:
                self.current_transitions += 1 / video_chunk_remain

            self.bit_rate = self.algo_instance.select_representation(
                probabilities, self.current_transitions
            )

        return (state, reward, end_of_video, data, recommended_rates)
