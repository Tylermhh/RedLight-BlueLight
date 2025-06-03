import cv2
import numpy as np
import time
from ultralytics import YOLO #install in anaconda 

# Load the YOLO pose estimation model
model = YOLO("yolo11n-pose.pt")

# COCO pose keypoint connections (skeleton)
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # legs
]

# COCO keypoint names for reference
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


class PoseTracker:
    def __init__(self, movement_threshold=50, max_distance=100):
        """
        Initialize pose tracker

        Args:
            movement_threshold: Minimum pixel distance to consider as movement
            max_distance: Maximum distance to associate poses between frames
        """
        self.poses = {}  # Dictionary to store tracked poses
        self.next_id = 0  # Counter for assigning new IDs
        self.movement_threshold = movement_threshold
        self.max_distance = max_distance
        self.last_check_time = time.time()
        self.check_interval = 0.1  # Check for movement every second

    def calculate_pose_center(self, keypoints, confidence_threshold=0.5):
        """Calculate the center point of a pose based on visible keypoints"""
        valid_points = []
        for kpt in keypoints:
            if len(kpt) >= 3 and kpt[2] > confidence_threshold:
                valid_points.append([kpt[0], kpt[1]])

        if valid_points:
            return np.mean(valid_points, axis=0)
        return None

    def calculate_pose_distance(self, pose1, pose2, confidence_threshold=0.5):
        """Calculate distance between two poses using visible keypoints"""
        valid_points1 = []
        valid_points2 = []

        for i, (kpt1, kpt2) in enumerate(zip(pose1, pose2)):
            if (len(kpt1) >= 3 and kpt1[2] > confidence_threshold and
                    len(kpt2) >= 3 and kpt2[2] > confidence_threshold):
                valid_points1.append([kpt1[0], kpt1[1]])
                valid_points2.append([kpt2[0], kpt2[1]])

        if len(valid_points1) < 3:  # Need at least 3 points for reliable comparison
            return float('inf')

        # Calculate average distance between corresponding keypoints
        distances = []
        for p1, p2 in zip(valid_points1, valid_points2):
            distances.append(np.linalg.norm(np.array(p1) - np.array(p2)))

        return np.mean(distances)

    def update_poses(self, new_keypoints_list):
        """Update tracked poses with new detections"""
        current_time = time.time()
        new_poses = {}

        if not new_keypoints_list:
            # No poses detected, keep existing poses but mark as lost
            for pose_id, pose_data in self.poses.items():
                pose_data['last_seen'] = current_time
                pose_data['status'] = 'lost'
            return

        # Convert new keypoints to numpy arrays
        new_poses_data = []
        for keypoints in new_keypoints_list:
            if hasattr(keypoints, 'cpu'):
                keypoints = keypoints.cpu().numpy()
            new_poses_data.append(keypoints)

        # If no existing poses, assign new IDs to all
        if not self.poses:
            for i, pose_data in enumerate(new_poses_data):
                new_poses[self.next_id] = {
                    'keypoints': pose_data,
                    'last_keypoints': pose_data.copy(),
                    'last_seen': current_time,
                    'center': self.calculate_pose_center(pose_data),
                    'status': 'active',
                    'last_movement_check': current_time
                }
                self.next_id += 1
        else:
            # Match new poses with existing ones
            existing_ids = list(self.poses.keys())
            existing_poses = [self.poses[pid]['keypoints'] for pid in existing_ids]

            # Calculate distance matrix
            if existing_poses:
                distances = []
                for new_pose in new_poses_data:
                    pose_distances = []
                    for existing_pose in existing_poses:
                        dist = self.calculate_pose_distance(new_pose, existing_pose)
                        pose_distances.append(dist)
                    distances.append(pose_distances)

                distances = np.array(distances)

                # Assign poses using greedy matching
                used_existing = set()
                used_new = set()

                for _ in range(min(len(new_poses_data), len(existing_ids))):
                    if distances.size == 0:
                        break

                    # Find minimum distance
                    min_pos = np.unravel_index(np.argmin(distances), distances.shape)
                    new_idx, existing_idx = min_pos

                    if (distances[min_pos] < self.max_distance and
                            new_idx not in used_new and existing_idx not in used_existing):
                        # Match found
                        pose_id = existing_ids[existing_idx]
                        old_keypoints = self.poses[pose_id]['keypoints'].copy()

                        new_poses[pose_id] = {
                            'keypoints': new_poses_data[new_idx],
                            'last_keypoints': old_keypoints,
                            'last_seen': current_time,
                            'center': self.calculate_pose_center(new_poses_data[new_idx]),
                            'status': 'active',
                            'last_movement_check': self.poses[pose_id].get('last_movement_check', current_time)
                        }

                        used_new.add(new_idx)
                        used_existing.add(existing_idx)

                    # Remove this combination from consideration
                    distances[new_idx, :] = float('inf')
                    distances[:, existing_idx] = float('inf')

                # Add unmatched new poses as new tracks
                for i, pose_data in enumerate(new_poses_data):
                    if i not in used_new:
                        new_poses[self.next_id] = {
                            'keypoints': pose_data,
                            'last_keypoints': pose_data.copy(),
                            'last_seen': current_time,
                            'center': self.calculate_pose_center(pose_data),
                            'status': 'active',
                            'last_movement_check': current_time
                        }
                        self.next_id += 1

        self.poses = new_poses

    def check_movement(self):
        """Check for movement in tracked poses and return list of moved pose IDs"""
        current_time = time.time()

        if current_time - self.last_check_time < self.check_interval:
            return []

        moved_poses = []

        for pose_id, pose_data in self.poses.items():
            if pose_data['status'] != 'active':
                continue

            # Check if enough time has passed since last movement check
            if current_time - pose_data['last_movement_check'] >= self.check_interval:
                current_keypoints = pose_data['keypoints']
                last_keypoints = pose_data['last_keypoints']

                # Calculate movement distance
                movement_distance = self.calculate_pose_distance(current_keypoints, last_keypoints)

                if movement_distance > self.movement_threshold:
                    moved_poses.append(pose_id)
                    print(f"Movement detected for pose ID {pose_id}: {movement_distance:.1f} pixels")

                # Update last movement check
                pose_data['last_movement_check'] = current_time
                pose_data['last_keypoints'] = current_keypoints.copy()

        self.last_check_time = current_time
        return moved_poses

    def get_active_poses(self):
        """Return list of currently active poses"""
        return {pid: data for pid, data in self.poses.items() if data['status'] == 'active'}


def draw_pose(frame, keypoints, pose_id, confidence_threshold=0.5):
    """Draw pose keypoints and skeleton on the frame with ID label"""
    if keypoints is None or len(keypoints) == 0:
        return frame

    # Convert to numpy if it's a tensor
    if hasattr(keypoints, 'cpu'):
        keypoints = keypoints.cpu().numpy()

    h, w = frame.shape[:2]

    # Draw skeleton connections
    for connection in POSE_CONNECTIONS:
        pt1_idx, pt2_idx = connection

        if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
            pt1 = keypoints[pt1_idx]
            pt2 = keypoints[pt2_idx]

            # Check if both points have sufficient confidence
            if len(pt1) >= 3 and len(pt2) >= 3:
                if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                    # Convert to pixel coordinates
                    x1, y1 = int(pt1[0]), int(pt1[1])
                    x2, y2 = int(pt2[0]), int(pt2[1])

                    # Draw line between keypoints
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw keypoints
    for i, keypoint in enumerate(keypoints):
        if len(keypoint) >= 3 and keypoint[2] > confidence_threshold:
            x, y = int(keypoint[0]), int(keypoint[1])

            # Different colors for different body parts
            if i < 5:  # head keypoints
                color = (255, 0, 0)  # blue
            elif i < 11:  # arm keypoints
                color = (0, 255, 0)  # green
            else:  # leg keypoints
                color = (0, 0, 255)  # red

            # Draw keypoint circle
            cv2.circle(frame, (x, y), 4, color, -1)
            cv2.circle(frame, (x, y), 6, (255, 255, 255), 2)

    # Draw pose ID label
    if len(keypoints) > 0:
        # Find a good position for the label (use nose if available, otherwise first valid keypoint)
        label_pos = None
        if len(keypoints[0]) >= 3 and keypoints[0][2] > confidence_threshold:  # nose
            label_pos = (int(keypoints[0][0]), int(keypoints[0][1]) - 20)
        else:
            for keypoint in keypoints:
                if len(keypoint) >= 3 and keypoint[2] > confidence_threshold:
                    label_pos = (int(keypoint[0]), int(keypoint[1]) - 20)
                    break

        if label_pos:
            cv2.putText(frame, f"ID: {pose_id}", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return frame


def main():
    # Initialize webcam and pose tracker
    cap = cv2.VideoCapture(0)
    tracker = PoseTracker(movement_threshold=5, max_distance=250)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set camera properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting pose tracking... Press 'q' to quit")
    print("Movement detection runs every second")

    try:
        # Get predictions from the model
        results = model(0, stream=True, verbose=False)

        for result in results:
            # Get the original frame
            frame = result.orig_img

            # Process keypoints and update tracker
            keypoints_list = []
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                # Extract keypoints data (includes x, y, confidence)
                for kpts_data in result.keypoints.data:
                    keypoints_list.append(kpts_data)

            # Update pose tracker
            tracker.update_poses(keypoints_list)

            # Check for movement
            moved_poses = tracker.check_movement()
            if moved_poses:
                print(f"Poses that moved: {moved_poses}")

            # Draw all active poses
            active_poses = tracker.get_active_poses()
            for pose_id, pose_data in active_poses.items():
                frame = draw_pose(frame, pose_data['keypoints'], pose_id)

            # Add text overlay with tracking info
            cv2.putText(frame, f"Tracked poses: {len(active_poses)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Press 'q' to quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display recent movement info
            if moved_poses:
                movement_text = f"Movement: IDs {moved_poses}"
                cv2.putText(frame, movement_text,
                            (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Display the frame
            cv2.imshow('Pose Tracking', frame)

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

        # Print final tracking summary
        active_poses = tracker.get_active_poses()
        print(f"\nFinal summary:")
        print(f"Total poses tracked: {len(active_poses)}")
        for pose_id, pose_data in active_poses.items():
            print(f"  Pose ID {pose_id}: {pose_data['status']}")


if __name__ == "__main__":
    main()