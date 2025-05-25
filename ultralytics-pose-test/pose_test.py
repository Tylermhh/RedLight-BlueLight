import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO pose estimation model
model = YOLO("yolo11n-pose.pt")

# COCO pose keypoint connections (skeleton)
# These define which keypoints to connect with lines
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


def draw_pose(frame, keypoints, confidence_threshold=0.5):
    """
    Draw pose keypoints and skeleton on the frame

    Args:
        frame: OpenCV frame
        keypoints: YOLO keypoints tensor (17, 3) - x, y, confidence
        confidence_threshold: minimum confidence to draw keypoint
    """
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

    return frame


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set camera properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting pose estimation... Press 'q' to quit")

    try:
        # Get predictions from the model
        results = model(0, stream=True, verbose=False)

        for result in results:
            # Get the original frame
            frame = result.orig_img

            # Process each detected person
            if result.keypoints is not None:
                keypoints_list = result.keypoints.xy  # Get x,y coordinates
                confidence_list = result.keypoints.conf if hasattr(result.keypoints, 'conf') else None

                # Draw pose for each detected person
                for i, keypoints in enumerate(keypoints_list):
                    # Combine x,y with confidence if available
                    if confidence_list is not None and i < len(confidence_list):
                        # Stack x, y, confidence
                        kpts_with_conf = np.column_stack([
                            keypoints.cpu().numpy(),
                            confidence_list[i].cpu().numpy()
                        ])
                    else:
                        # Use keypoints data which includes confidence
                        kpts_with_conf = result.keypoints.data[i].cpu().numpy()

                    frame = draw_pose(frame, kpts_with_conf)

            # Add text overlay
            cv2.putText(frame, f"Pose Estimation - Press 'q' to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display the frame
            cv2.imshow('Pose Estimation', frame)

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()