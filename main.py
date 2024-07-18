from utils import (read_video, 
                   save_video)
from trackers import PlayerTracker

def main():
    # Read video
    input_video_path = f"input_videos/input_video.mp4"
    output_video_path = f"output_videos/output_video.avi"

    video_frames = read_video(input_video_path)
    
    # Detect players
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(video_frames)

    # vvv Draw output vvv
    # Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)


    save_video(output_video_frames, output_video_path)


if __name__ == "__main__":
    main()
