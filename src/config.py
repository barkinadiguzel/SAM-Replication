# Model parameters
INPUT_SIZE = (3, 224, 224)    # RGB images, 224x224
NUM_CLASSES = 1000            # Example: ImageNet
USE_RELU = True               # Optional ReLU activation

# SAM parameters
RHO = 0.05                    # Perturbation magnitude for SAM

# Optimizer parameters
BASE_OPTIMIZER = "SGD"        # Options: "SGD" or "Adam"
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
