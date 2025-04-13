from constants import *
from random import random


def get_green_light_time_seconds() -> float:
    return random() * GREEN_LIGHT_TIME_RANGE_SECONDS + GREEN_LIGHT_TIME_MIN_SECONDS


class Game:
    def __init__(self):
        self.state: GameState = GameState.READ_FACES
        self.face_filters_playing: list = []
        self.face_filters_won: list = []
        self.face_filters_lost: list = []

    def run(self):
        # game loop skeleton
        while self.state != GameState.END_GAME:
            match self.state:
                case GameState.READ_FACES:
                    self.read_faces()
                case GameState.GREEN_LIGHT:
                    self.green_light()
                case GameState.RED_LIGHT:
                    self.red_light()

        # end game
        self.end_game()

    def read_faces(self):
        # initialize contents of self.face_filters_playing with each player
        pass

        # start red light / green light loop
        self.state = GameState.GREEN_LIGHT

    def green_light(self):
        # stay green for a random amount of time
        # time is predetermined when green light state is first entered
        pass

        # if no players left playing, end game
        # otherwise, go to red light state
        if not self.face_filters_playing:
            self.state = GameState.END_GAME
        else:
            self.state = GameState.RED_LIGHT

    def red_light(self):
        # stay red for defined amount of time (RED_LIGHT_TIME_SECONDS)
        # check for movement during this time
        pass

        # if no players left playing, end game
        # otherwise, go to green light state
        if not self.face_filters_playing:
            self.state = GameState.END_GAME
        else:
            self.state = GameState.GREEN_LIGHT

    def end_game(self):
        # report results
        pass


def main():
    game = Game()
    game.run()

if __name__ == '__main__':
    main()