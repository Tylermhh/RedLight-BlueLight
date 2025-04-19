from enum import Enum, auto


RED_LIGHT_TIME_SECONDS: int = 5
GREEN_LIGHT_TIME_MIN_SECONDS: int = 5
GREEN_LIGHT_TIME_RANGE_SECONDS: int = 5

class GameState(Enum):
    READ_FACES = auto()
    GREEN_LIGHT = auto()
    RED_LIGHT = auto()
    END_GAME = auto()


class Color(Enum):
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)


STATE_COLORS: dict[GameState, Color] = {
    GameState.READ_FACES: Color.WHITE,
    GameState.GREEN_LIGHT: Color.GREEN,
    GameState.RED_LIGHT: Color.RED,
    GameState.END_GAME: Color.WHITE
}