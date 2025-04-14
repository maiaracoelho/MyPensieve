RESEVOIR = 4  # BB - 5
CUSHION = 8  # BB - 10


def bb_algo(buffer_size, bitrates, DEFAULT_QUALITY, M_IN_K):
    # print("bb_algo")
    if buffer_size < RESEVOIR:
        bit_rate = DEFAULT_QUALITY
    elif buffer_size >= RESEVOIR + CUSHION:
        bit_rate = len(bitrates) - 1
    else:
        bit_rate = int((len(bitrates) - 1) * (buffer_size - RESEVOIR) / float(CUSHION))

    # print(bit_rate)

    return bit_rate
