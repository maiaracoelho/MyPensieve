import numpy as np
import math

qoe_option = "qoeCost"
pesos = [0.40, 0.25, 0.15, 0.20]
pesos1 = [0.50, 0.50]

REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1.0
SEXP = 0.5
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

    def calculate_cost(self, data):
        segment_size = float(data["video_chunk_size"])
        bit_rate = data["bit_rate"]
        max_bit_rate = int(data["max_bit_rate"])
        max_segment_size = float(np.max(data["next_video_chunk_sizes"]))
        maxCost = (
            float(COST[max_bit_rate]["c"] + COST[max_bit_rate]["j"]) * max_segment_size
        )
        cost = (COST[bit_rate]["c"] + COST[bit_rate]["j"]) * segment_size

        totalCost = 1.0 - (float(cost) / float(maxCost))
        return totalCost

    def calculate_qoer(self, data):
        utility = self.calculate_bitrate_average(data["bit_rate"])  / float(data["max_bit_rate"])
        rebuf_index = 1.0 - (data["rebufering_time"] / self.bMin)
        amplitude_index = 1.0 - float(
            abs(data["bit_rate"] - data["last_bit_rate"]) / float(self.rMax - self.rMin)
        )
        delayInSec = data["delay"] / 1000.0
        delay_index = 1.0 - float(abs(SEXP - delayInSec) / max(SEXP, delayInSec))

        qoeR = (
            pesos[0] * utility
            + pesos[1] * (rebuf_index)
            + pesos[2] * (amplitude_index)
            + pesos[3] * (delay_index)
        )
        return qoeR

    def calculate_qoep(self, data):
        bitrate = data["bit_rate"]
        rebuffering = data["rebufering_time"]
        last_bit_rate = data["last_bit_rate"]

        reward = (
            float(bitrate)/1000.0
            - REBUF_PENALTY * float(rebuffering)
            - SMOOTH_PENALTY * float(np.abs(float(bitrate) - float(last_bit_rate)))/1000.0
        )
        return reward

    def calculate_qoeCost(self, data):
        rew = float(
            pesos1[0] * self.calculate_qoer(data)
            + pesos1[1] * self.calculate_cost(data)
        )
        return rew

    def calculate_reward(self, data):
        if qoe_option == "qoep":
            return self.calculate_qoep(data)
        elif qoe_option == "qoer":
            return self.calculate_qoer(data)
        elif qoe_option == "qoeCost":
            return self.calculate_qoeCost(data)
        else:
            return self.calculate_cost(data)
