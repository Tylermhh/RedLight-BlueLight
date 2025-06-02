from player import Player
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from pose_test import PoseTracker, draw_pose
from ultralytics import YOLO

# Load MoveNet MultiPose model
movenet = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet_input_size = 256

#yolo pose
pose_model = YOLO("yolo11n-pose.pt")
tracker = PoseTracker(movement_threshold=5, max_distance=250)

def preprocess_frame(frame):
    img = cv2.resize(frame, (movenet_input_size, movenet_input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.int32)
    return img[np.newaxis, ...]

def extract_faces_from_keypoints(frame, people, threshold=0.3):
    faces = []
    draw_boxes = []
    h, w, _ = frame.shape

    for person in people[0]:
        keypoints = np.reshape(person[:51], (17, 3))

        # Fixed issue with detecting extra people since MoveNet is looking for 6 people
        high_confidence_points = [kp for kp in keypoints if kp[2] >= threshold]
        if len(high_confidence_points) < 6:
            continue

        x = int(keypoints[0][1] * w)
        y = int(keypoints[0][0] * h)
        face_size = 80
        x1, y1 = max(0, x - face_size // 2), max(0, y - face_size // 2)
        x2, y2 = min(w, x + face_size // 2), min(h, y + face_size // 2)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (100, 100))
        faces.append(face_resized)
        draw_boxes.append((x1, y1, x2, y2))

    return faces, draw_boxes

def get_player_filters() -> dict[int, Player]:
    cap = cv2.VideoCapture(0)
    players = {}
    next_id = 1

    print("Initializing players... Press 'c' when all players are in frame.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Press 'c' to capture players", frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            input_img = preprocess_frame(frame)
            outputs = movenet.signatures["serving_default"](tf.constant(input_img))
            people = outputs["output_0"].numpy()

            faces, boxes = extract_faces_from_keypoints(frame, people)

            for i in range(len(faces)):
                face = faces[i]
                x1, y1, x2, y2 = boxes[i]

                players[next_id] = Player(id=next_id, face_filter=face)
                print(f"Player_{next_id} initialized")

                # Draw box and ID on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Player {next_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                next_id += 1

            # Show result for a second
            cv2.imshow("Detected Faces", frame)
            cv2.waitKey(1000)
            break

    cap.release()
    cv2.destroyAllWindows()
    return players

#check_player_movement:
# --- During red light, grabs a camera frame
# --- Detects players in the frame
# --- Crops their current face regions
# --- Calls is_moving(current_face) on each player
# --- If the face changed significantly, they’re eliminated

def check_player_movement(players_playing: dict[int, Player], players_lost: dict[int, Player]) -> None:
    # Capture a frame from the webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read from webcam during red light.")
        return
    
    results = pose_model(frame, stream=False, verbose=False)
    keypoints_list = []
    if results and results[0].keypoints is not None:
        for kpts_data in results[0].keypoints.data:
            keypoints_list.append(kpts_data.cpu().numpy() if hasattr(kpts_data, 'cpu') else kpts_data)

    # Update the tracker with all detected poses
    tracker.update_poses(keypoints_list)

    # Ask the tracker which pose IDs moved
    moved_pose_ids = tracker.check_movement()
    if moved_pose_ids:
        print(f"Detected movement in pose IDs: {moved_pose_ids}")


    to_eliminate = []
    for player_id in players_playing:
        pose_id = player_id - 1  # <-- adjust if your mapping differs
        if pose_id in moved_pose_ids:
            to_eliminate.append(player_id)

    for pid in to_eliminate:
        print("Player {} moved".format(pid))
        players_lost[pid] = players_playing.pop(pid)
        
    '''
    input_img = preprocess_frame(frame)
    outputs = movenet.signatures["serving_default"](tf.constant(input_img))
    people = outputs["output_0"].numpy()
    _, boxes = extract_faces_from_keypoints(frame, people)

    players_caught_ids: list[int] = []

    for i, player_id in enumerate(list(players_playing.keys())):
        if i >= len(boxes):
            continue  # Not enough faces detected

        x1, y1, x2, y2 = boxes[i]
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (100, 100))

        if players_playing[player_id].is_moving(face_resized):
            print(f"Player {player_id} moved! Eliminating...")
            players_caught_ids.append(player_id)

    for player_id in players_caught_ids:
        players_lost[player_id] = players_playing.pop(player_id)'''

def check_player_winning(players_playing: dict[int, Player], players_won: dict[int, Player]) -> None:
    # TODO: interface with CV script to check if players have won the game
    # this is a stub
    # how to do this???
    players_won_ids: list[int] = []
    for player_id in players_playing:
        if players_playing[player_id].is_won():
            players_won_ids.append(player_id)

    for player_id in players_won_ids:
        players_won[player_id] = players_playing.pop(player_id)