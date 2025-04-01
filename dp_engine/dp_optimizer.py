import tensorflow as tf

def dynamic_noise(sensitivity, epoch, max_epochs=10, decay_schedule="exponential"):
    """
    Compute a decaying noise multiplier.
    
    For "exponential", returns sensitivity * (0.98^epoch), with a lower bound of 0.5.
    For "linear", returns sensitivity - (sensitivity-0.5)*(epoch/max_epochs), with a lower bound of 0.5.
    """
    if decay_schedule == "exponential":
        return max(sensitivity * (0.98 ** epoch), 0.5)
    elif decay_schedule == "linear":
        return max(sensitivity - (sensitivity - 0.5) * (epoch / max_epochs), 0.5)
    else:
        return max(sensitivity * (0.98 ** epoch), 0.5)

def compute_epsilon(num_examples, batch_size, noise_multiplier, epochs, delta=1e-5):
    """
    Compute epsilon using TensorFlow Privacy's RDP accountant (with default orders).
    
    Returns:
        (epsilon, optimal_order)
    """
    try:
        from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
        eps, optimal_order = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
            n=num_examples,
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
            epochs=epochs,
            delta=delta
        )
        return eps, optimal_order
    except ImportError:
        print("TensorFlow Privacy not installed. Returning epsilon as None.")
        return None, None

class DPAdam:
    def __init__(self, learning_rate=0.005, sensitivity=0.7, l2_norm_clip=1.5,
                 noise_multiplier=0.42, num_microbatches=32, decay_schedule="exponential"):
        self.learning_rate = float(learning_rate)
        self.sensitivity = float(sensitivity)
        self.l2_norm_clip = float(l2_norm_clip)
        self.base_noise_multiplier = float(noise_multiplier)
        self.num_microbatches = int(num_microbatches)
        self.decay_schedule = decay_schedule
        # Use the legacy Adam optimizer (helpful on Apple M1/M2)
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
    
    def initialize_optimizer(self, epoch):
        # Placeholder; no per-epoch reinitialization is needed.
        pass
    
    def get_optimizer(self):
        return self.optimizer

@tf.function(reduce_retracing=True, experimental_relax_shapes=True)
def compute_dp_gradients_eager(tape, loss, variables, l2_norm_clip, noise_multiplier):
    """
    Compute per-example gradients with clipping and add Gaussian noise.
    
    - Uses tf.vectorized_map to compute gradients for each example.
    - Replaces any None gradients with zeros.
    - Clips the gradients using L2-norm and adds noise.
    
    Returns: List of aggregated gradients (same shape as corresponding variable).
    """
    batch_size = tf.shape(loss)[0]
    per_example_grads = []
    for var in variables:
        def grad_fn(i):
            g = tape.gradient(loss[i], var)
            return g if g is not None else tf.zeros_like(var)
        per_example_grad = tf.vectorized_map(grad_fn, tf.range(batch_size))
        per_example_grads.append(per_example_grad)
    
    clipped_grads = []
    for g in per_example_grads:
        g_flat = tf.reshape(g, [batch_size, -1])
        norms = tf.norm(g_flat, axis=1)
        clip_factors = tf.minimum(1.0, l2_norm_clip / (norms + 1e-6))
        r = tf.rank(g)
        reshape_dims = tf.concat([[batch_size], tf.ones([r - 1], dtype=tf.int32)], axis=0)
        g_clipped = g * tf.reshape(clip_factors, reshape_dims)
        clipped_grads.append(g_clipped)
    
    aggregated_grads = []
    batch_size_float = tf.cast(batch_size, tf.float32)
    for g in clipped_grads:
        avg_grad = tf.reduce_sum(g, axis=0) / batch_size_float
        noise_stddev = l2_norm_clip * noise_multiplier / batch_size_float
        noise = tf.random.normal(tf.shape(avg_grad), stddev=noise_stddev)
        aggregated_grads.append(avg_grad + noise)
    
    return aggregated_grads
