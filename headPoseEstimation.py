import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import threading

def speak_async(text):
    def run():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run).start()

engine = pyttsx3.init()
wasGazing = False
# Eye landmark indices (left and right eye)
LEFT_EYE = [33, 133, 160, 158, 153, 144, 145, 23]
RIGHT_EYE = [263, 362, 385, 387, 373, 380, 374, 253]
IRIS_LEFT = [468, 469, 470, 471]  # Approximate iris points
IRIS_RIGHT = [473, 474, 475, 476]

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5,
                                  max_num_faces=5,
                                 refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
color_tuple = (0, 255, 0)
drawing_spec = mp_drawing.DrawingSpec(color=color_tuple, thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

# TODO: compensate for angle with respect to camera
# TODO: light up face when cheating (details TBD)
# TODO: Associate cheaters with student entities (and thus names)
# TODO: Call out names
# TODO: make array of students

def get_mesh_results(image):
    """
    get the face meshes from the video frame
    :param image: frame from video to be processed
    :return: set of mesh objects (0-max_num_faces of faces)
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Performance optimization

    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def get_eye_direction(landmarks, img_w, img_h):
    """ Determines eye gaze direction by tracking iris position """
    def get_valid_points(indices):
        """Returns valid landmark points"""
        return [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in indices if i < len(landmarks)]
    # print(len(landmarks), max(IRIS_RIGHT))
    if len(landmarks) < max(IRIS_RIGHT) + 1:
        return "No valid eye data"
    
    left_eye_points = get_valid_points(LEFT_EYE)
    right_eye_points = get_valid_points(RIGHT_EYE)
    iris_left = get_valid_points(IRIS_LEFT)
    iris_right = get_valid_points(IRIS_RIGHT)

    if len(left_eye_points) < 3 or len(right_eye_points) < 3 or len(iris_left) < 2 or len(iris_right) < 2:
        return "No valid eye data"
    left_eye_points = np.array([(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in LEFT_EYE])
    right_eye_points = np.array([(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in RIGHT_EYE])
    iris_left = np.array([(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in IRIS_LEFT])
    iris_right = np.array([(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in IRIS_RIGHT])
    
    # Get the middle points of the eye and iris
    left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
    right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
    left_iris_center = np.mean(iris_left, axis=0).astype(int)
    right_iris_center = np.mean(iris_right, axis=0).astype(int)
    
    
    if left_iris_center[0] < left_eye_center[0] - 4 and right_iris_center[0] < right_eye_center[0] - 4 or left_iris_center[0] > left_eye_center[0] + 5 and right_iris_center[0] > right_eye_center[0] + 5:
        return True
    
    return False

def adjust_pos(horizontal, vertical, x_pos):
    return horizontal, vertical

def get_direction(horizontal, vertical, eyes_gazing, x_pos):
    hor_adj, vert_adj = adjust_pos(horizontal, vertical, x_pos)


    text = ""
    global drawing_spec
    global color_tuple
    if eyes_gazing or (hor_adj < -3.5 or hor_adj > 3.5):
        text = "Gazing"
        speak_async("Gazing detected")

    if vert_adj < -3:
        text += " Down"
    elif vert_adj > 3:
        text += " Up"

    if not text:
        text = "Straight"
    if "Gazing" in text:
        drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
        color_tuple = (0, 0, 255)
    else:
        drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        color_tuple = (0, 255, 0)

    return text

def writeValue(image, p1, p2, horizontal, vertical, text, x_pos):
    cv2.line(image, p1, p2, color_tuple, 3)
    cv2.putText(image, text, (x_pos, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color_tuple, 2)
    cv2.putText(image, f"horizontal: {horizontal:.2f}", (x_pos, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color_tuple, 2)
    cv2.putText(image, f"vertical: {vertical:.2f}", (x_pos, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_tuple, 2)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    start = time.time()

    results = get_mesh_results(image)

    img_h, img_w, _ = image.shape  # Get image dimensions

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_2d = []  # Stores 2D points
            face_3d = []  # Stores 3D points
            nose_2d, nose_3d = None, None

            # Select important landmarks for pose estimation
            important_landmarks = [33, 263, 1, 61, 291, 199]
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in important_landmarks:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    if idx == 1:  # Nose landmark
                        nose_2d = (x, y)
                        nose_3d = (x, y, lm.z * 3000)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            # Convert lists to NumPy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera matrix and distortion coefficients
            focal_length = img_w
            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                   [0, focal_length, img_h / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP to get rotation and translation vectors
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Convert rotation vector to rotation matrix
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            # -------------------------------------------

            # Get the face bounding box or use key landmarks
            face_center_x = (face_landmarks.landmark[33].x + face_landmarks.landmark[263].x) / 2
            face_center_y = (face_landmarks.landmark[1].y + face_landmarks.landmark[152].y) / 2

            # Normalize position relative to camera center
            image_width = image.shape[1]
            image_height = image.shape[0]

            # Normalize to range [-1, 1] where 0 is center of the frame
            x_offset = (face_center_x - (image_width / 2)) / (image_width / 2)
            y_offset = (face_center_y - (image_height / 2)) / (image_height / 2)

            # Scale based on expected FOV correction
            horizontal_correction = x_offset * 15  # Adjust for field of view (≈15° per half-screen width)
            vertical_correction = y_offset * 10  # Adjust for FOV (≈10° per half-screen height)

            # -------------------------------------------

            # Extract head orientation angles
            vertical = angles[0] * 360
            horizontal = angles[1] * 360
            # Determine head position
            eyes_gazing = get_eye_direction(face_landmarks.landmark, img_w, img_h)
            #print(eyes_gazing)


            # Project nose direction onto 2D
            nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + horizontal * 10), int(nose_2d[1] - vertical * 10))

            # text = get_direction(horizontal_correction, vertical_correction, eyes_gazing, p1[0])
            text = get_direction(horizontal, vertical, eyes_gazing, p1[0])


            writeValue(image, p1, p2, horizontal, vertical, text, p1[0])


            # Draw face landmarks
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    # Calculate FPS
    end = time.time()
    fps = 1 / (end - start)
    cv2.putText(image, f"FPS: {fps:.2f}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show output
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Exit on 'Esc' key
        break

cap.release()
cv2.destroyAllWindows()
