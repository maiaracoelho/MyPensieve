import numpy as np

qoe_option = "qoep"
pesos = [0.40, 0.25, 0.15, 0.20]

REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
SMOOTH_PENALTY = 1
BMIN = 6.0
class RewardMetrics():

    def __init__(self):

        self.bit_rate = 0
        self.last_bit_rate = 0
        self.buffer_size = 0
        self.bitrate_list = []
        
    def calculate_bitrate_average(self, bit_rate):
        self.bitrate_list.append(bit_rate)
        return sum(self.bitrate_list) / len(self.bitrate_list)

    def calculate_qoer(self, data):
    
        max_bitrate = data['max_bitrate']
        bitrate_average = self.calculate_bitrate_average(data['bit_rate'])
        utility = bitrate_average / float(max_bitrate)
        prob = data[]
        rebuf_index = prob

        qoeR = (
        pesos[0] * utility
        + pesos[0] * (1.0 - amplitude_index)
        + pesos[0] * (1.0 - rebuf_index)
        + pesos[0] * (1 - delay_index)
        )

        return qoeR


    def calculate_qoep(data):
        bitrate = data['bit_rate']
        rebuffering = data['rebufering_time']
        last_bit_rate = data['last_bit_rate']
    
        reward = (
        float(bitrate)
        - REBUF_PENALTY * float(rebuffering)
        - SMOOTH_PENALTY * float(np.abs(float(bitrate) - float(last_bit_rate)))
        )
        return reward


def calculate_reward(self, data):
    reward = self.calculate_qoep(data) if (qoe_option == "qoep") else self.calculate_qoer(data)
    return reward
