import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers

def build_ehr_model(input_dim, num_classes):
    """
    Build a deep neural network for EHR classification.
    
    Architecture:
      - Dense layer with 256 units and 'swish' activation (with L2 regularization 0.01)
      - BatchNormalization (256 units)
      - Dropout (0.4)
      - Dense layer with 128 units and 'swish' activation (with L2 regularization 0.01)
      - BatchNormalization (128 units)
      - Dropout (0.3)
      - Dense layer with num_classes units and softmax activation
    """
    model = tf.keras.Sequential([
        Dense(256, activation='swish', input_dim=input_dim,
              kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='swish', kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model
