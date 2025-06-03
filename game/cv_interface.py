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

        # Only count person if enough confident keypoints
        high_confidence_points = [kp for kp in keypoints if kp[2] >= threshold]
        if len(high_confidence_points) < 6:
            continue

        x = int(keypoints[0][1] * w)
        y = int(keypoints[0][0] * h)

        face_size = 80
        x1, y1 = max(0, x - face_size // 2), max(0, y - face_size // 2)
        x2, y2 = min(w, x + face_size // 2), min(h, y + face_size // 2)

        # FIXED NUM OF PEOPLE by filtering overlapping boxes
        duplicate = False
        for (ex1, ey1, ex2, ey2) in draw_boxes:
            iou = compute_iou((x1, y1, x2, y2), (ex1, ey1, ex2, ey2))
            if iou > 0.4:
                duplicate = True
                break
        if duplicate:
            continue

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (100, 100))
        faces.append(face_resized)
        draw_boxes.append((x1, y1, x2, y2))

    return faces, draw_boxes

#IoU stands for Intersection over Union - using it to make sure MoveNet doesn't double count the same person
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def get_player_filters(cap) -> dict[int, Player]:
    players = {}
    next_id = 1

    print("Initializing players... Press 'c' when all players are in frame.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] Failed to read frame from webcam.")
            return {}

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

    cv2.destroyAllWindows()
    return players

def check_player_movement(cap, players_playing: dict[int, Player], players_lost: dict[int, Player]) -> None:
    ret, frame = cap.read()

    if not ret or frame is None:
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
    # This is a stub
    players_won_ids: list[int] = []
    for player_id in players_playing:
        if players_playing[player_id].is_won():
            players_won_ids.append(player_id)

    for player_id in players_won_ids:
        players_won[player_id] = players_playing.pop(player_id)
