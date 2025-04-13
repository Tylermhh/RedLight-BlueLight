from enum import Enum

RED_LIGHT_TIME_SECONDS: int = 5

class GameState(Enum):
    READ_FACES = 0
    GREEN_LIGHT = 1
    RED_LIGHT = 2