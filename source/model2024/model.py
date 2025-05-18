def create_branch1_model(input_shape, embedding_dim=64, dropout_rate=0.2):
    """Create the LSTM-based model for Branch 1 (large-scale data).
    
    Architecture:
    1. Input layer (timesteps, features)
    2. Two LSTM layers for temporal pattern extraction
    3. Batch normalization and dropout for regularization
    4. Dense layer for final embedding
    
    Args:
        input_shape: Shape of the input data (timesteps, features)
        embedding_dim: Dimension of the embedding space
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Keras model for Branch 1
        
    Note:
        - Uses forward-only LSTM layers suitable for forecasting
        - Includes batch normalization and dropout for regularization
        - Simple architecture focused on temporal pattern extraction
    """
    inputs = layers.Input(shape=input_shape, name='branch1_input')
    
    # First LSTM layer with return sequences
    x = layers.LSTM(embedding_dim, return_sequences=True, 
                    name='branch1_lstm1')(inputs)
    x = layers.BatchNormalization(name='branch1_bn1')(x)
    x = layers.Dropout(dropout_rate, name='branch1_dropout1')(x)
    
    # Second LSTM layer
    x = layers.LSTM(embedding_dim, return_sequences=False, 
                    name='branch1_lstm2')(x)
    x = layers.BatchNormalization(name='branch1_bn2')(x)
    x = layers.Dropout(dropout_rate, name='branch1_dropout2')(x)
    
    # Dense layer to create embedding
    outputs = layers.Dense(embedding_dim, activation='relu', 
                          name='branch1_embedding')(x)
    
    return Model(inputs=inputs, outputs=outputs, name='branch1_model') 