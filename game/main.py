from constants import *
from player import Player
from datetime import datetime, timedelta
from random import random
import cv2

# Import updated cv_interface functions directly
from cv_interface import get_player_filters, check_player_movement, check_player_winning

class Game:
    def __init__(self):
        self.state: GameState = GameState.READ_FACES
        self.players_playing: dict[int, Player] = {}
        self.players_won: dict[int, Player] = {}
        self.players_lost: dict[int, Player] = {}
        self.time_for_next_state = datetime.now()
        self.cap = None  # Camera will be initialized in run()

    def run(self) -> None:
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("❌ Could not open video capture")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("❌ Failed to grab frame")
                break

            match self.state:
                case GameState.READ_FACES:
                    self.read_faces()
                case GameState.GREEN_LIGHT:
                    self.green_light()
                case GameState.RED_LIGHT:
                    self.red_light()
                case GameState.END_GAME:
                    self.end_game()
                    break

            border_color = STATE_COLORS[self.state]
            height, width = frame.shape[:2]

            # draw border and text
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), border_color.value, thickness=10)
            cv2.putText(frame, f"State: {self.state.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, border_color.value, 2)

            if self.state in [GameState.RED_LIGHT, GameState.GREEN_LIGHT]:
                time_remaining_str = str(self.time_for_next_state - datetime.now())
                cv2.putText(frame, f"{time_remaining_str[time_remaining_str.index(':') + 1:]}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, border_color.value, 2)

            cv2.putText(frame, f"Players: {len(self.players_playing)}", (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, border_color.value, 2)

            cv2.imshow('Live Feed', frame)

            # Temp key listeners for quitting or ending
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                print("🔚 Forcing end game...")
                self.state = GameState.END_GAME

        self.cap.release()
        cv2.destroyAllWindows()

    def read_faces(self) -> None:
        print("🔍 Reading player filters...")
        self.players_playing = get_player_filters(self.cap)
        print(f"✅ Loaded {len(self.players_playing)} player(s)")
        self.to_green_light_state()

    def green_light(self) -> None:
        if self.time_for_next_state < datetime.now() and self.players_playing:
            self.to_red_light_state()
            return
        else:
            check_player_winning(self.players_playing, self.players_won)

        if not self.players_playing:
            self.state = GameState.END_GAME

    def red_light(self) -> None:
        if self.time_for_next_state < datetime.now() and self.players_playing:
            self.to_green_light_state()
            return
        else:
            check_player_movement(self.cap, self.players_playing, self.players_lost)
            check_player_winning(self.players_playing, self.players_won)

        if not self.players_playing:
            self.state = GameState.END_GAME
            print(f"🎮 Transitioned to state: {self.state}")

    def end_game(self):
        print("🎉 Game over!")
        print(f"🏁 Winners: {[p.id for p in self.players_won.values()]}")
        print(f"❌ Eliminated: {[p.id for p in self.players_lost.values()]}")
        print("Press 'q' to exit...")
        
        while True:
            key = input()
            if key.lower() == 'q':
                break

    def to_green_light_state(self) -> None:
        self.set_time_for_next_state_green()
        self.state = GameState.GREEN_LIGHT
        print(f"🎮 Transitioned to state: {self.state}")

    def to_red_light_state(self) -> None:
        self.set_time_for_next_state_red()
        self.state = GameState.RED_LIGHT
        print(f"🎮 Transitioned to state: {self.state}")

    def set_time_for_next_state_green(self) -> None:
        duration = random() * GREEN_LIGHT_TIME_RANGE_SECONDS + GREEN_LIGHT_TIME_MIN_SECONDS
        self.time_for_next_state = datetime.now() + timedelta(seconds=duration)

    def set_time_for_next_state_red(self) -> None:
        self.time_for_next_state = datetime.now() + timedelta(seconds=RED_LIGHT_TIME_SECONDS)


def main() -> None:
    game = Game()
    game.run()


if __name__ == '__main__':
    main()
