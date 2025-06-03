from constants import *
from cv_interface import *
from random import random
from datetime import datetime, timedelta
from pose_test import PoseTracker, draw_pose
from ultralytics import YOLO
import cv2


class Game:
    def __init__(self):
        self.state: GameState = GameState.READ_FACES
        self.players_playing: dict[int, Player] = {}
        self.players_won: dict[int, Player] = {}
        self.players_lost: dict[int, Player] = {}
        self.time_for_next_state = datetime.now()
        self.cap = None  # for error with webcam

        # yolo pose instead
        self.pose_model = YOLO("yolo11n-pose.pt")
        self.tracker = PoseTracker(movement_threshold=5, max_distance=250)

    def run(self) -> None:
        self.read_faces()

        import time
        time.sleep(0.5)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("error with camear")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                time.sleep(0.5)
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    break
                else:
                    continue

            results = self.pose_model(frame, stream=False, verbose=False)
            keypoints_list = []
            if results and results[0].keypoints is not None:
                for kpts_data in results[0].keypoints.data:
                    keypoints_list.append(kpts_data.cpu().numpy() if hasattr(kpts_data, 'cpu') else kpts_data)

            self.tracker.update_poses(keypoints_list)

            # Detect moved poses and eliminate corresponding Player IDs
            moved_pose_ids = self.tracker.check_movement()
            if moved_pose_ids:
                to_eliminate = []
                for player_id in self.players_playing:
                    pose_id = player_id - 1
                    if pose_id in moved_pose_ids:
                        to_eliminate.append(player_id)
                for pid in to_eliminate:
                    print(f"Player {pid} moved! Eliminating…")
                    self.players_lost[pid] = self.players_playing.pop(pid)

            # adding pose 
            active = self.tracker.get_active_poses()
            for pid, pose_data in active.items():
                frame = draw_pose(frame, pose_data['keypoints'], pid)

            match self.state:
                case GameState.GREEN_LIGHT:
                    self.green_light()
                case GameState.RED_LIGHT:
                    self.red_light()
                case GameState.END_GAME:
                    self.end_game()

            border_color = STATE_COLORS[self.state]
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color.value, thickness=10)
            cv2.putText(
                frame,
                f"State: {self.state.name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                border_color.value,
                2
            )


            cv2.putText(
                frame,
                f"Players: {len(self.players_playing)}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                border_color.value,
                2
            )

            cv2.imshow('Live Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def read_faces(self) -> None:
        self.players_playing = get_player_filters()
        self.to_green_light_state()

    def green_light(self) -> None:
        if datetime.now() >= self.time_for_next_state and self.players_playing:
            self.to_red_light_state()
            return
        else:
            check_player_winning(self.players_playing, self.players_won)

        if not self.players_playing:
            self.state = GameState.END_GAME

    def red_light(self) -> None:
        if datetime.now() >= self.time_for_next_state and self.players_playing:
            self.to_green_light_state()
            return
        else:
            check_player_winning(self.players_playing, self.players_won)

        if not self.players_playing:
            self.state = GameState.END_GAME

    def end_game(self):
        pass

    def to_green_light_state(self) -> None:
        green_light_time_seconds = random() * GREEN_LIGHT_TIME_RANGE_SECONDS + GREEN_LIGHT_TIME_MIN_SECONDS
        self.time_for_next_state = datetime.now() + timedelta(seconds=green_light_time_seconds)
        self.state = GameState.GREEN_LIGHT

    def to_red_light_state(self) -> None:
        self.time_for_next_state = datetime.now() + timedelta(seconds=RED_LIGHT_TIME_SECONDS)
        self.state = GameState.RED_LIGHT


def main() -> None:
    game = Game()
    game.run()


if __name__ == '__main__':
    main()
