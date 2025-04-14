import numpy as np


class Stallion:
    def __init__(
        self,
        video_bit_rate,
        window_size=8,
        z_thr=0.3,
        z_latency=0.75,
        lat_threshold=1.5,
        min_stable_steps=3,
    ):
        """
        :param video_bit_rate: Lista de bitrates disponíveis (em Kbps).
        :param window_size: Tamanho da janela deslizante.
        :param z_thr: Fator de sensibilidade ao desvio padrão do throughput.
        :param z_latency: Fator de sensibilidade à latência.
        :param lat_threshold: Limiar de latência segura (em segundos).
        :param min_stable_steps: Quantidade mínima de ciclos com alta estimativa para subir de qualidade.
        """
        self.video_bit_rate = video_bit_rate
        self.window_size = window_size
        self.z_thr = z_thr
        self.z_latency = z_latency
        self.lat_threshold = lat_threshold
        self.min_stable_steps = min_stable_steps

        self.last_quality = 1
        self.thr_window = []
        self.lat_window = []
        self.stable_counter = 0

    def update_metrics(self, throughput_kbps, latency_s):
        if len(self.thr_window) >= self.window_size:
            self.thr_window.pop(0)
        if len(self.lat_window) >= self.window_size:
            self.lat_window.pop(0)

        self.thr_window.append(throughput_kbps)
        self.lat_window.append(latency_s)

    def select_quality(self):
        if not self.thr_window:
            return self.last_quality  # Sem dados ainda

        avg_thr = np.mean(self.thr_window)
        std_thr = np.std(self.thr_window)

        # Bitrate seguro com base na fórmula multiplicativa (como no artigo)
        if avg_thr > 0:
            safe_thr = avg_thr * (1 - self.z_thr * (std_thr / avg_thr))
        else:
            safe_thr = 0.0

        # Penalização por latência
        avg_lat = np.mean(self.lat_window)
        std_lat = np.std(self.lat_window)
        safe_lat = avg_lat + self.z_latency * std_lat

        if safe_lat > self.lat_threshold:
            penalty = safe_lat / self.lat_threshold
            safe_thr /= penalty  # Reduz safe_thr se latência for alta

        # Escolhe a maior qualidade possível dentro do safe_thr
        chosen_quality = 0
        for i in reversed(range(len(self.video_bit_rate))):
            if self.video_bit_rate[i] <= safe_thr:
                chosen_quality = i
                break

        # Estabilidade: evita upgrades muito rápidos
        if chosen_quality > self.last_quality:
            self.stable_counter += 1
            if self.stable_counter >= self.min_stable_steps:
                self.last_quality = chosen_quality
                self.stable_counter = 0
        elif chosen_quality < self.last_quality:
            self.last_quality = chosen_quality
            self.stable_counter = 0
        else:
            # mesma qualidade
            self.stable_counter = 0

        return self.last_quality
