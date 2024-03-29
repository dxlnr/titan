"""MuZero Configurations"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Conf:
    """Defining default params.

    :param NUM_ACTORS: Number of players in the game.
    :param OBSERVATION_SHAPE: Dimensions of the game observation, must be 3D (height, width, channel).
    :param ACTION_SHAPE: Defines the dimension of the action space.
        In chess, 8 planes are used to encode the action.
        The first one-hot plane encodes which position the piece was moved from.
        The next two planes encode which position the piece was moved to:
            A one-hot plane to encode the target position, if on the board,
            and a second binary plane to indicate whether the target was valid (on the board) or not.
        The remaining five binary planes are used to indicate the type of promotion,
        if any (queen, knight, bishop, rook, none).

    :param ACTION_SPACE: NRayDirs x MaxRayLength + NKnightDirs + NPawnDirs * NMinorPromotions,
        encoding a probability distribution over 64 x 73 = 4,672 possible moves.
    :param CHANNELS: Number of hidden planes for each convolution. Default stated in paper
        is 256.
    """
    # GAME
    NUM_ACTORS: int = 2
    OBSERVATION_SHAPE: Tuple[int] = (119, 8, 8)
    ACTION_SHAPE: Tuple[int] = (8, 8, 8)
    ACTION_SPACE: list = field(default_factory=list)

    # MODEL
    CHANNELS: int = 256
    DEPTH: int = 16
    REDUCED_C_REWARD: int = 256
    REDUCED_C_VALUE: int = 256
    REDUCED_C_POLICY: int = 256
    RESNET_FC_REWARD_LAYERS: Tuple[int] = (256, 256)
    RESNET_FC_VALUE_LAYERS: Tuple[int] = (256, 256)
    RESNET_FC_POLICY_LAYERS: Tuple[int] = (256, 256)
    SUPPORT_SIZE: int = 10

    # MCTS
    MAX_MOVES: int = 512
    NUM_ROLLOUTS: int = 200
    DISCOUNT: float = 1.0
    EPSILON: float = 0.001
    ROOT_DIRICHLET_ALPHA: float = 0.3
    ROOT_EXPLORATION_FRACTION: float = 0.25
    PB_C_BASE: int = 19652
    PB_C_INIT: int = 1.25

    # If we already have some information about which values occur in the
    # environment, we can use them to initialize the rescaling.
    # This is not strictly necessary, but establishes identical behaviour to
    # AlphaZero in board games.
    # KNOWN_BOUNDS

    # TRAINING 
    LOSS: str = "mse"
    TRAINING_STEPS: int = int(1e5)
    CHECKPOINT_INTERVAL: int = int(1e3)
    WINDOW_SIZE: int = int(1e6)
    BATCH_SIZE: int = 2048
    NUM_UNROLL_STEPS: int = 5
    TD_STEPS: int = 10
    WEIGHT_DECAY: float = 1e-4
    MOMENTUM: float = 0.9
    LR_INIT: float = 1e-3
    LR_DECAY_RATE = 0.1
    LR_DECAY_STEPS: int = 1e5

    # DATA 
    MODEL_PATH: str = ''

    def __init__(self):
        self.ACTION_SPACE = list(range(64 * 73))

    def __post_init__(self):
        pass

    def __getitem__(self, item):
        return getattr(self, item)

    def __repr__(self):
        rs = "Configurations:\n"
        rs += "".join(
            f"  {k}: {self[k]}\n" if k != "ACTION_SPACE" else ""
            for k in self.__dataclass_fields__.keys()
        )
        return rs
