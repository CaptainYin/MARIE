from enum import Enum


class Env(str, Enum):
    STARCRAFT = "starcraft"
    PETTINGZOO = "pettingzoo"
    GRF = "football"
    MAMUJOCO = "mamujoco"
    BIDEXHANDS = "bidexhands"
    SMAX = "smax"
    SMACv2 = "SMACv2"

RANDOM_SEED = 23
