import torch
from torchvision import models, transforms
import cv2



class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=False)
        # replace last layer
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(model_path, map_location=device))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # predict only first image because camera is not moving
    def predict(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            
        keypoints = outputs.squeeze().cpu().numpy()
        # map it to original image height and width
        original_h, original_w = img_rgb.shape[:2]
        # Adjust x cordinates
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints
    
    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            # int because pixel position can't be fraction
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            # display number of keypoint and dot
            cv2.putText(image, str(i//2), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            cv2.circle(image, (x,y), 8, (255, 0, 255), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames

