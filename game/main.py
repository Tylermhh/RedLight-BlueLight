from constants import *
from random import random
from datetime import datetime, timedelta


class Game:
    def __init__(self):
        self.state: GameState = GameState.READ_FACES
        self.face_filters_playing: list = [] # list of what?
        self.face_filters_won: list = [] # list of what?
        self.face_filters_lost: list = [] # list of what?
        self.time_for_next_state = datetime.now()

    def run(self):
        # TODO: capture and show camera feed
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
        # TODO: initialize contents of self.face_filters_playing with each player
        pass

        # start red light / green light loop
        self.to_green_light_state()

    def green_light(self):
        # stay green for a random amount of time
        # time is predetermined when green light state is first entered
        if self.time_for_next_state < datetime.now() and self.face_filters_playing:
            # if there are still players left playing, go to red light state
            self.to_red_light_state()

        # if no players left playing, end game
        if not self.face_filters_playing:
            self.state = GameState.END_GAME

    def red_light(self):
        # stay red for defined amount of time (RED_LIGHT_TIME_SECONDS)
        # check for movement during this time
        if self.time_for_next_state < datetime.now() and self.face_filters_playing:
            # if there are still players left playing, go to green light state
            self.to_green_light_state()
        else:
            # TODO: check for movement each frame
            pass

        # if no players left playing, end game
        if not self.face_filters_playing:
            self.state = GameState.END_GAME

    def end_game(self):
        # TODO: report results
        pass

    def to_green_light_state(self):
        self.set_time_for_next_state_green()
        self.state = GameState.GREEN_LIGHT

    def to_red_light_state(self):
        self.set_time_for_next_state_red()
        self.state = GameState.RED_LIGHT

    def set_time_for_next_state_green(self):
        green_light_time_seconds = random() * GREEN_LIGHT_TIME_RANGE_SECONDS + (
                 GREEN_LIGHT_TIME_MIN_SECONDS)
        self.time_for_next_state = (
                datetime.now() + timedelta(seconds=green_light_time_seconds)
        )

    def set_time_for_next_state_red(self):
        self.time_for_next_state = (
            datetime.now() + timedelta(seconds=RED_LIGHT_TIME_SECONDS)
        )


def main():
    game = Game()
    game.run()

if __name__ == '__main__':
    main()