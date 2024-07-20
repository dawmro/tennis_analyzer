from utils import (read_video, 
                   save_video)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2



def main():
    # Read video
    input_video_path = f"input_videos/input_video.mp4"
    output_video_path = f"output_videos/output_video.avi"

    video_frames = read_video(input_video_path)
    
    # Detect players
    player_tracker = PlayerTracker(model_path='models/yolov8x.pt')
    player_detections = player_tracker.detect_frames(frames=video_frames,
                                                     read_from_stub=True,
                                                     stub_path=f"tracker_stubs/player_detections.pkl")
    
    # Detect ball
    ball_tracker = BallTracker(model_path='models/yolo5_best.pt')
    ball_detections = ball_tracker.detect_frames(frames=video_frames,
                                                     read_from_stub=True,
                                                     stub_path=f"tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    
    # Detect court lines
    court_model_path = f"models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # vvv Draw output vvv
    # Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    # Draw ball bounding boxes
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)
    # Draw court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(video_frames, court_keypoints)

    # draw frame number
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 255, 0), 2)


    save_video(output_video_frames, output_video_path)


if __name__ == "__main__":
    main()
