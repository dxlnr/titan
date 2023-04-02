"""Shared Storage"""
from titan.config import Conf
from titan.models import M0Net


class SharedStorage:
    """Network Wrapper."""

    def __init__(self, config: Conf):
        self._model = M0Net(config)

    def get_latest_model(self) -> M0Net:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            pass
            # policy -> uniform, value -> 0, reward -> 0
            # return make_uniform_network()

    def save_model(self, step: int, model: M0Net) -> None:
        """Save torch model."""
        pass
        # self._networks[step] = network
