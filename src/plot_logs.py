import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "/app/results"
GRAPHS_DIR = "/app/graphs"

if not os.path.exists(GRAPHS_DIR):
    os.makedirs(GRAPHS_DIR)

for log_file in os.listdir(RESULTS_DIR):
    log_path = os.path.join(RESULTS_DIR, log_file)

    try:
        data = pd.read_csv(
            log_path,
            names=[
                "time_stamp",
                "bit_rate",
                "buffer_size",
                "rebuffer_time",
                "chunk_size",
                "download_time",
                "throughput",
            ],
            sep=",",
            engine="python",
        )
    except Exception as e:
        print(f"Erro ao ler {log_file}: {e}")
        continue

    if (
        "time_stamp" not in data.columns
        or "bit_rate" not in data.columns
        or "buffer_size" not in data.columns
    ):
        print(f"Arquivo {log_file} está no formato incorreto.")
        continue

    plt.figure()
    plt.plot(data["time_stamp"], data["bit_rate"], label="Bitrate (Kbps)")
    plt.xlabel("Time (s)")
    plt.ylabel("Bitrate (Kbps)")
    plt.title(f"Bitrate over Time - {log_file}")
    plt.legend()
    bitrate_graph_path = os.path.join(GRAPHS_DIR, f"{log_file}_bitrate.png")
    plt.savefig(bitrate_graph_path)
    plt.close()

    plt.figure()
    plt.plot(
        data["time_stamp"],
        data["buffer_size"],
        label="Buffer Size (s)",
        color="orange",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Buffer Size (s)")
    plt.title(f"Buffer Size over Time - {log_file}")
    plt.legend()
    buffer_graph_path = os.path.join(GRAPHS_DIR, f"{log_file}_buffer.png")
    print(buffer_graph_path)
    plt.savefig(buffer_graph_path)
    plt.close()

print("Gráficos gerados com sucesso na pasta 'graphs'.")
