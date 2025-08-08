from ultralytics import YOLO
from deepface import DeepFace
import cv2
import mediapipe as mp

#함수1 : 인원수 ====================================================================================

def get_num_people(image_path):
    model = YOLO('yolov8n.pt')
    results = model(image_path)
    return len([obj for obj in results[0].boxes.cls if int(obj) == 0])



#함수2 : 성비 ====================================================================================

def get_gender_distribution(image_path):
    analysis = DeepFace.analyze(img_path=image_path, actions=['gender'])
    genders = {'male': 0, 'female': 0}
    for face in analysis:
        if face['gender'] == 'Man':
            genders['male'] += 1
        else:
            genders['female'] += 1
    return genders



#함수3 : 포즈 ====================================================================================

def get_pose_type(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    image = cv2.imread(image_path)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
            return "hands_up"
        elif (left_wrist.y < left_shoulder.y + 0.1) and (right_wrist.y < right_shoulder.y + 0.1):
            return "peace_sign"
        else:
            return "other_pose"
    else:
        return "no_person"