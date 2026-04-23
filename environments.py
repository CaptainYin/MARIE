from enum import Enum


class Env(str, Enum):
    STARCRAFT = "starcraft"
    PETTINGZOO = "pettingzoo"
    GRF = "football"
    MAMUJOCO = "mamujoco"
    BIDEXHANDS = "bidexhands"
    SMAX = "smax"

RANDOM_SEED = 23
