"""
Definition of the CNN model architecture for RFID classification
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def build_cnn_model(input_shape, num_classes, config, recognition_mode='id'):
    """
    Builds the CNN model for classification
    
    Args:
        input_shape: Shape of input data (channels, points)
        num_classes: Number of classes to predict
        config: Model configuration
        recognition_mode: Recognition mode ('id', 'distance', 'height', 'angle', or 'multi')
        
    Returns:
        model: Compiled CNN model
    """
    # Model parameters
    conv_filters = config['conv_filters']
    conv_kernel_sizes = config['conv_kernel_sizes']
    pool_sizes = config['pool_sizes']
    dropout_rate = config['dropout_rate']
    dense_layers = config['dense_layers']
    
    # CRITICAL FIX: Handle angle recognition mode differently
    # In angle mode, data comes as (points, channels) instead of (channels, points)
    if recognition_mode == 'angle':
        # For angle recognition, skip the adapter transformation
        tf_input_shape = input_shape  # Keep original (201, 2) shape
    else:
        # Original adapter for other modes
        tf_input_shape = (input_shape[1], input_shape[0])
    
    # Construire le modèle de base
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    
    input_layer = Input(shape=tf_input_shape)
    
    # Convolutional layers
    x = input_layer
    for i, (filters, kernel_size, pool_size) in enumerate(zip(conv_filters, conv_kernel_sizes, pool_sizes)):
        x = Conv1D(filters, kernel_size, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        
        # SAFETY CHECK: Only apply pooling if dimension is large enough
        current_dim = x.shape[1]
        if current_dim is not None and current_dim >= pool_size:
            x = MaxPooling1D(pool_size=pool_size)(x)
        else:
            print(f"Skipping pooling with size {pool_size} for dimension {current_dim}")
    
    # Flatten layer
    x = Flatten()(x)
    
    # Shared dense layers
    for units in dense_layers:
        x = Dense(units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    
    # Output layer based on recognition mode
    if recognition_mode == 'id':
        # Classification ID standard
        output = Dense(len(num_classes), activation='softmax', name='output')(x)
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    elif recognition_mode == 'distance':
        if isinstance(num_classes, tuple):
            # Régression pour les distances continues (num_classes est (min_dist, max_dist))
            output = Dense(1, activation='linear', name='distance_output')(x)
            model = Model(inputs=input_layer, outputs=output)
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae', 'mse']
            )
        else:
            # Classification pour distances discrètes (num_classes est le nombre de classes)
            output = Dense(len(num_classes), activation='softmax', name='output')(x)
            model = Model(inputs=input_layer, outputs=output)
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
    
    elif recognition_mode == 'height':
        if isinstance(num_classes, tuple):
            # Régression pour les hauteurs continues (num_classes est (min_height, max_height))
            output = Dense(1, activation='linear', name='height_output')(x)
            model = Model(inputs=input_layer, outputs=output)
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae', 'mse']
            )
        else:
            # Classification pour hauteurs discrètes (num_classes est le nombre de classes)
            output = Dense(len(num_classes), activation='softmax', name='output')(x)
            model = Model(inputs=input_layer, outputs=output)
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
    
    # Mode height_class: Classification binaire pour h1/h2 (académiquement valide)
    elif recognition_mode == 'height_class':
        from config import HEIGHT_CLASS_NAMES
        output = Dense(2, activation='softmax', name='height_class_output')(x)
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    elif recognition_mode == 'multi':
        # Classification multi-tâche (ID et distance)
        num_id_classes, num_dist_classes = num_classes
        
        if isinstance(num_dist_classes, tuple):
            # ID classification + distance regression
            id_output = Dense(len(num_id_classes), activation='softmax', name='id_output')(x)
            dist_output = Dense(1, activation='linear', name='distance_output')(x)
            
            model = Model(inputs=input_layer, outputs=[id_output, dist_output])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss={
                    'id_output': 'sparse_categorical_crossentropy',
                    'distance_output': 'mse'
                },
                metrics={
                    'id_output': ['accuracy'],
                    'distance_output': ['mae', 'mse']
                },
                loss_weights={
                    'id_output': 0.5,
                    'distance_output': 0.5
                }
            )
        else:
            # ID classification + distance classification
            id_output = Dense(len(num_id_classes), activation='softmax', name='id_output')(x)
            dist_output = Dense(len(num_dist_classes), activation='softmax', name='distance_output')(x)
            
            model = Model(inputs=input_layer, outputs=[id_output, dist_output])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss={
                    'id_output': 'sparse_categorical_crossentropy',
                    'distance_output': 'sparse_categorical_crossentropy'
                },
                metrics={
                    'id_output': ['accuracy'],
                    'distance_output': ['accuracy']
                },
                loss_weights={
                    'id_output': 0.5,
                    'distance_output': 0.5
                }
            )
    
    # Ajout du support pour la reconnaissance d'angle
    elif recognition_mode == 'angle':
        # Classification pour les angles (-30°, 0°, 30°)
        output = Dense(len(num_classes), activation='softmax', name='angle_output')(x)
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    else:
        # Fallback: treat as classification
        output = Dense(num_classes if isinstance(num_classes, int) else len(num_classes), activation='softmax', name='output')(x)
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def get_callbacks(model_path, patience=10, recognition_mode='id'):
    """
    Defines callbacks for training
    
    Args:
        model_path (str): Path where to save the model
        patience (int): Patience for early stopping
        recognition_mode (str): Recognition mode ('id', 'distance', 'height', 'angle', or 'multi')
        
    Returns:
        list: List of callbacks
    """
    # Determine the best metric to monitor based on recognition mode
    if recognition_mode in ['distance', 'height']:
        checkpoint_monitor = 'val_loss'  # Use val_loss for regression tasks
        checkpoint_mode = 'min'  # Lower is better for loss metrics
    elif recognition_mode == 'multi':
        checkpoint_monitor = 'val_loss'  # Use overall val_loss for multi-task
        checkpoint_mode = 'min'  # Lower is better for loss metrics
    else:  # 'id' or default
        checkpoint_monitor = 'val_accuracy'  # Use accuracy for classification
        checkpoint_mode = 'max'  # Higher is better for accuracy
    
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor=checkpoint_monitor,
            mode=checkpoint_mode,
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=0.00001,
            verbose=1
        )
    ]