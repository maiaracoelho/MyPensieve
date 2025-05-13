import numpy as np
import math

pesos = [0.20, 0.25, 0.15, 0.40]
pesos1 = [0.50, 0.50]
ALPHA = 0.7
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1.0
SEXP = 0.5
BETA_DELAY = 2.0
GAMMA_REBUF = 0.5
COST = {
    300: {"c": 0.1, "j": 0.05},
    750: {"c": 0.2, "j": 0.1},
    1200: {"c": 0.4, "j": 0.2},
    1850: {"c": 0.6, "j": 0.3},
    2850: {"c": 0.8, "j": 0.4},
    4300: {"c": 0.95, "j": 0.55},
}


class RewardMetrics:
    def __init__(self, bMin, rMin, rMax):
        self.bMin = bMin
        self.rMin = rMin
        self.rMax = rMax
        self.bitrate_list = []

    def calculate_bitrate_average(self, bit_rate):
        self.bitrate_list.append(bit_rate)
        return sum(self.bitrate_list) / len(self.bitrate_list)

    def calculate_cost(self, data,scen=None):
        segment_size = float(data["video_chunk_size"])
        bit_rate = data["bit_rate"]
        next_video_chunk_sizes = data["next_video_chunk_sizes"]
        cost = 0.0
        if scen=="edge":
            for i, rate in enumerate(COST.keys()):
                seg_size_i = next_video_chunk_sizes[i]
                cost += (COST[rate]["c"] + COST[rate]["j"]) * seg_size_i
        elif scen=="cloud":
            cost = (COST[bit_rate]["c"] + COST[bit_rate]["j"]) * segment_size
        else:
            cost = (COST[bit_rate]["c"] + COST[bit_rate]["j"]) * segment_size

        maxCost = 0.0
        for i, rate in enumerate(COST.keys()):
            seg_size_i = next_video_chunk_sizes[i]
            maxCost += (COST[rate]["c"] + COST[rate]["j"]) * seg_size_i

        if maxCost < 1e-8:
            return 0.0

        C_trans = cost / maxCost
        C_trans = np.clip(C_trans, 0.0, 1.0)
        return C_trans

    def calculate_rebuf_index(self, rebuffering_time):
        return np.exp(-GAMMA_REBUF * rebuffering_time)

    def calculate_amplitude_index(self, bit_rate, last_bit_rate):
        return np.clip(
            1.0 - abs(bit_rate - last_bit_rate) / float(self.rMax - self.rMin),
            0.0,
            1.0,
        )

    def calculate_delay_index(self, delay_ms):
        delay_in_sec = delay_ms / 1000.0
        if delay_in_sec <= SEXP:
            return 1.0
        else:
            return np.exp(-BETA_DELAY * (delay_in_sec - SEXP))

    def calculate_qoer(self, data):
        utility = self.calculate_bitrate_average(data["bit_rate"]) / float(self.rMax)
        rebuf_index = self.calculate_rebuf_index(data["rebufering_time"])
        amplitude_index = self.calculate_amplitude_index(data["bit_rate"], data["last_bit_rate"])
        delay_index = self.calculate_delay_index(data["delay"])

        qoeR = (
            pesos[0] * utility
            + pesos[1] * rebuf_index
            + pesos[2] * amplitude_index
            + pesos[3] * delay_index
        )

        return qoeR

    def calculate_qoep(self, data):
        bitrate = data["bit_rate"]
        rebuffering = data["rebufering_time"]
        last_bit_rate = data["last_bit_rate"]
        return (
            float(bitrate) / 1000.0
            - REBUF_PENALTY * float(rebuffering)
            - SMOOTH_PENALTY * float(np.abs(float(bitrate) - float(last_bit_rate))) / 1000.0
        )

    def calculate_qoeCost(self, data, scen=None):
        qoe = self.calculate_qoer(data)
        cost = self.calculate_cost(data, scen)
        qoeCost = ALPHA * qoe + (1 - ALPHA) * (1 - cost)
        return qoeCost

    def calculate_reward(self, data, mode, scen=None):
        if mode == "qoep":
            return self.calculate_qoep(data)

        elif mode == "qoer":
            return self.calculate_qoer(data)

        elif mode == "qoeCost":
            return self.calculate_qoeCost(data,scen)

        else:
            return self.calculate_cost(data, scen)
