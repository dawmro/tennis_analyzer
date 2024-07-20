from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append(".")
from utils import get_center_of_bbox, measure_distance


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)


    def choose_and_filter_players(self, court_keypoints, player_detections):
        # choose players based on a first frame
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame) 
        # filter detection only to two chosen players
        filtered_player_detections = []
        for player_dict in player_detections:
            # loop over each track_in in player dict and pick only the ones that are in chosen players
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}      
            filtered_player_detections.append(filtered_player_dict) 

        return filtered_player_detections

    
    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            # calculate distance between player and each point of the court
            min_distance = float("inf")
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # sort based on minimum distance and choose two track_ids with the lowest value
        distances.sort(key = lambda x: x[1])
        chosen_players = [distances[0][0], distances[1][0]]

        return chosen_players


    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        # read and reuse intermediate data
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        # create list of detected frames
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        # save intermediate data to reuse during next run
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections


    def detect_frame(self, frame):
        # persist tracking within all frames
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        # key: player id; value: bounding box
        player_dict = {}
        
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            # need object class name
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            # want to select only people
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict
    

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # draw bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] -10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames





        