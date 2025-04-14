class Lolypop:
    def __init__(self, sigma_star, omega_star, DEFAULT_QUALITY):
        self.sigma_star = sigma_star
        self.omega_star = omega_star
        self.last_quality = DEFAULT_QUALITY  # Qualidade do último segmento

    def select_representation(self, probabilities, current_transitions):
        """
        Seleciona a qualidade para o próximo segmento com base no LOLYPOP.
        """
        # Passo 1: Identificar a qualidade mais alta que atende Σ*
        max_quality = 0
        for j, p in enumerate(probabilities):
            if p >= 1 - self.sigma_star:
                max_quality = j

        # Passo 2: Verificar o limite Ω* para transições de qualidade
        if current_transitions > self.omega_star and max_quality > self.last_quality:
            selected_quality = self.last_quality
        else:
            selected_quality = max_quality

        self.last_quality = selected_quality
        return selected_quality
