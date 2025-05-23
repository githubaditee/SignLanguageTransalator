import cv2
import mediapipe as mp
import time
import os
import numpy as np
from matplotlib import pyplot as plt

#Initial model
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable=False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image,results):
    mp_draw.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_draw.draw_landmarks(image, results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
    mp_draw.draw_landmarks(image, results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(image, results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image,results):
    mp_draw.draw_landmarks(image, results.face_landmarks,mp_holistic.FACEMESH_TESSELATION, mp_draw.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=1),
                           mp_draw.DrawingSpec(color=(80,256,121),thickness=1, circle_radius=1))
    mp_draw.draw_landmarks(image, results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec(color=(80,22,10),thickness=2,circle_radius=4),
                           mp_draw.DrawingSpec(color=(80,44,121),thickness=2, circle_radius=2))
    mp_draw.draw_landmarks(image, results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                           mp_draw.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4),
                           mp_draw.DrawingSpec(color=(121,44,250),thickness=2, circle_radius=2))
    mp_draw.draw_landmarks(image, results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                           mp_draw.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4),
                           mp_draw.DrawingSpec(color=(245,66,230),thickness=2, circle_radius=2))
    
def extract_keypoints(results):
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flattern() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flattern() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flattern() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flattern() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lh,rh])


DATA_PATH = os.path.join("SL_data");
actions = np.array(["hello","Thank you","ILoveYou"])
no_Sequences = 30
sequence_length = 30

for action in actions:
    for sequence in range(no_Sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

colors = [(245,117,16),(117,245,16),(16,117,245)]
def prob_viz(res,actions,input_frame,colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame,(0,60+num*40),(int(prob*100) , 90+num*40), colors[num], -1)
        cv2.putText(output_frame,action[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

sequence=[]
sentence=[]
threshold=0.7
cap=cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_Sequences):
            for frame_num in range(sequence_length):

                while cap.isOpened():
                    ret, frame = cap.read()
                    image, results = mediapipe_detection(frame, holistic) 
                    print(results) 
            
                     # Draw landmarks 
                    draw_styled_landmarks(image, results)

                    #Apply collection logic
                if frame_num == 0:
                    cv2.putText(image, "STARTING COLLECTION",(120,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4,cv2.LINE_AA)
                    cv2.putText(image, "Collecting frames for {} video Number {}".format(action,sequence),(15,12)
                                ,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, "Collecting frames for {} video Number {}".format(action,sequence),(15,12),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                        
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH,action,str(sequence),str(frame_num))
                np.save(npy_path, keypoints)

                cv2.imshow('OpenCV Feed', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            #cap.release()
            #cv2.destroyAllWindows()

#plt.imshow(cv2.cvtColor((frame,results),cv2.COLOR_BGR2RGB))

