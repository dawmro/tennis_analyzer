from utils import (read_video, 
                   save_video)


def main():
    input_video_path = f"input_videos/input_video.mp4"
    output_video_path = f"output_videos/output_video.avi"

    video_frames = read_video(input_video_path)
    print(len(video_frames))

    save_video(video_frames, output_video_path)


if __name__ == "__main__":
    main()
