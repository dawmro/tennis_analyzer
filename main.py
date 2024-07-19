from utils import (read_video, 
                   save_video)
from trackers import PlayerTracker, BallTracker

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
                                                     read_from_stub=False,
                                                     stub_path=f"tracker_stubs/ball_detections.pkl")

    # vvv Draw output vvv
    # Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    # Draw ball bounding boxes
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)


    save_video(output_video_frames, output_video_path)


if __name__ == "__main__":
    main()
