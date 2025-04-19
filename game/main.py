from constants import *
from cv_interface import *
from random import random
from datetime import datetime, timedelta
import cv2


class Game:
    def __init__(self):
        self.state: GameState = GameState.READ_FACES
        self.players_playing: dict[int, Player] = {}
        self.players_won: dict[int, Player] = {}
        self.players_lost: dict[int, Player] = {}
        self.time_for_next_state = datetime.now()

    def run(self) -> None:
        cap = cv2.VideoCapture(0)  # Use 0 for default camera

        # game loop skeleton
        while True:
            # get frame
            ret, frame = cap.read()
            if not ret:
                break

            # perform action based on game state
            match self.state:
                case GameState.READ_FACES:
                    self.read_faces()
                case GameState.GREEN_LIGHT:
                    self.green_light()
                case GameState.RED_LIGHT:
                    self.red_light()
                case GameState.END_GAME:
                    # end game
                    self.end_game()

            # get color for current state
            border_color = STATE_COLORS[self.state]

            # draw border and text overlay
            height, width = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), border_color.value,
                          thickness=10)
            cv2.putText(frame, f"State: {self.state.name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, border_color.value, 2)

            # time remaining in *_Light states
            if self.state in [GameState.RED_LIGHT, GameState.GREEN_LIGHT]:
                time_remaining_str = str(self.time_for_next_state - datetime.now())
                cv2.putText(frame,
                            f"{time_remaining_str[time_remaining_str.index(':') + 1:]}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            border_color.value, 2)

            # display the frame
            cv2.imshow('Live Feed', frame)

            # exit on 'q' press
            # TODO: remove later
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # FOR TESTING
            # TODO: remove later
            if cv2.waitKey(1) & 0xFF == ord('e'):
                print('here')
                self.state = GameState.END_GAME

        # cleanup
        cap.release()
        cv2.destroyAllWindows()

    def read_faces(self) -> None:
        self.players_playing = get_player_filters()

        # start red light / green light loop
        self.to_green_light_state()

    def green_light(self) -> None:
        # stay green for a random amount of time
        # time is predetermined when green light state is first entered
        if self.time_for_next_state < datetime.now() and self.players_playing:
            # if there are still players left playing, go to red light state
            self.to_red_light_state()
            return
        else:
            check_player_winning(self.players_playing, self.players_won)

        # if no players left playing, end game
        if not self.players_playing:
            self.state = GameState.END_GAME

    def red_light(self) -> None:
        # stay red for defined amount of time (RED_LIGHT_TIME_SECONDS)
        # check for movement during this time
        if self.time_for_next_state < datetime.now() and self.players_playing:
            # if there are still players left playing, go to green light state
            self.to_green_light_state()
            return
        else:
            check_player_movement(self.players_playing, self.players_lost)
            check_player_winning(self.players_playing, self.players_won)

        # if no players left playing, end game
        if not self.players_playing:
            self.state = GameState.END_GAME

    def end_game(self):
        # TODO: report results
        pass

    def to_green_light_state(self) -> None:
        self.set_time_for_next_state_green()
        self.state = GameState.GREEN_LIGHT

    def to_red_light_state(self) -> None:
        self.set_time_for_next_state_red()
        self.state = GameState.RED_LIGHT

    def set_time_for_next_state_green(self) -> None:
        green_light_time_seconds = random() * GREEN_LIGHT_TIME_RANGE_SECONDS + (
                 GREEN_LIGHT_TIME_MIN_SECONDS)
        self.time_for_next_state = (
                datetime.now() + timedelta(seconds=green_light_time_seconds)
        )

    def set_time_for_next_state_red(self) -> None:
        self.time_for_next_state = (
            datetime.now() + timedelta(seconds=RED_LIGHT_TIME_SECONDS)
        )


def main() -> None:
    game = Game()
    game.run()

if __name__ == '__main__':
    main()