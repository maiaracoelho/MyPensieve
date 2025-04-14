import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def calculate_action_probabilities(action_prob, noise_std=0.01):
    """
    Calcula as probabilidades de ação com ruído e normalização.

    Args:
        action_prob (np.ndarray): Probabilidades iniciais de ação.
        noise_std (float): Desvio padrão do ruído normal.

    Returns:
        np.ndarray: Probabilidades ajustadas após adição de ruído e normalização.
    """
    assert not np.any(np.isnan(action_prob)), "action_prob contém valores NaN."
    assert np.all(action_prob >= 0), "action_prob contém valores negativos"
    assert np.all(action_prob <= 1), "action_prob contém valores maiores que 1"

    # Adicionar ruído gaussiano
    noise = np.random.gumbel(size=len(action_prob))
    # print("noise", noise)
    action_prob_with_noise = action_prob + noise
    action_prob_with_noise = np.clip(action_prob_with_noise, 1e-6, None)  # Evitar zeros
    action_prob_with_noise /= np.sum(action_prob_with_noise)  # Normalizar
    # print("action_prob_with_noise", action_prob_with_noise)

    # Aplicar softmax

    with tf.Session() as sess:
        try:
            noisy_action_prob = sess.run(tf.nn.softmax(action_prob_with_noise))
            # print("noisy_action_prob", noisy_action_prob)
        except Exception as e:
            print("Erro ao calcular noisy_action_prob:", e)
            noisy_action_prob = None

    assert noisy_action_prob is not None, "Erro ao calcular noisy_action_prob"
    assert not np.any(np.isnan(noisy_action_prob)), "noisy_action_prob contém NaN"
    assert np.all(noisy_action_prob >= 0), "noisy_action_prob contém valores negativos"

    return noisy_action_prob


def sample_actions_from_probabilities(noisy_action_prob):
    """
    Amostra ações com base nas probabilidades fornecidas, utilizando uma sessão do TensorFlow.

    Args:
        noisy_action_prob (np.ndarray): Probabilidades ajustadas de ação.

    Returns:Teste concluído.
        np.ndarray: Vetor binário de decisões.
    """
    # Verificar se as probabilidades são válidas
    assert np.all(noisy_action_prob >= 0), "noisy_action_prob contém valores negativos"
    assert np.all(
        noisy_action_prob <= 1
    ), "noisy_action_prob contém valores maiores que 1"

    # Criar distribuição binomial no TensorFlow
    noisy_action_prob_tensor = tf.constant(noisy_action_prob, dtype=tf.float32)
    binomial_dist = tfp.distributions.Binomial(
        total_count=1, probs=noisy_action_prob_tensor
    )
    decisions_tensor = binomial_dist.sample()

    # Executar a amostragem em uma sessão
    with tf.compat.v1.Session() as sess:
        decisions_eval = sess.run(decisions_tensor)

        if np.all(decisions_eval == 0):
            # print("Decisions zerado, setar aleatoriamente")
            random_index = np.random.choice(len(decisions_eval))
            decisions_eval[random_index] = 1
    # print("decisions_eval", decisions_eval)
    return decisions_eval
