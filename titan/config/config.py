"""MuZero Configurations"""
from typing import Tuple
from dataclasses import dataclass


@dataclass
class Conf:
    """Defining default params.

    :param NUM_ACTORS: Number of players in the game.
    :param OBSERVATION_SHAPE: Dimensions of the game observation, must be 3D (height, width, channel).
    :param SIZE_ACTION_SPACE: Defines the dimension of the action space.
        In chess, 8 planes are used to encode the action. 
        The first one-hot plane encodes which position the piece was moved from. 
        The next two planes encode which position the piece was moved to: 
            A one-hot plane to encode the target position, if on the board, 
            and a second binary plane to indicate whether the target was valid (on the board) or not. 
        The remaining five binary planes are used to indicate the type of promotion, 
        if any (queen, knight, bishop, rook, none).
    
    """
    # GAME
    NUM_ACTORS: int = 2
    OBSERVATION_SHAPE: Tuple[int] = (8, 8, 119)          
    SIZE_ACTION_SPACE: int = 8

    # MODEL 
    CHANNELS: int = 256 
    DEPTH: int = 16

    # VISIT_SOFTMAX_TEMPERATURE_FN
    MAX_MOVES: int = 1000
    NUM_SIMULATIONS: int = 1000
    DISCOUNT: float = 0.0

    # Root prior exploration noise.
    # ROOT_DIRICHLET_ALPHA
    ROOT_EXPLORATION_FRACTION: float = 0.25

    # UCB formula
    PB_C_BASE: int = 19652
    PB_C_INIT: int = 1.25

    # If we already have some information about which values occur in the
    # environment, we can use them to initialize the rescaling.
    # This is not strictly necessary, but establishes identical behaviour to
    # AlphaZero in board games.
    # KNOWN_BOUNDS

    ### Training
    TRAINING_STEPS: int = int(1e6)
    CHECKPOINT_INTERVAL: int = int(1e3)
    WINDOW_SIZE: int = int(1e6)
    BATCH_SIZE: int = 2048
    NUM_UNROLL_STEPS: int = 5
    TD_STEPS: int = 10

    WEIGHT_DECAY: float = 1e-4
    MOMENTUM: float = 0.9

    # Exponential learning rate schedule
    LR_INIT: float = 1e-3
    LR_DECAY_RATE = 0.1
    LR_DECAY_STEPS: int = 1e5

    def __init__(self):
        pass

    def __post_init__(self):
        pass
