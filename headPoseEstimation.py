import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5,
                                  max_num_faces=5)
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


def get_direction(horizontal, vertical):
    text = ""
    global drawing_spec
    global color_tuple
    if  horizontal < -3.5 or horizontal > 3.5:
        text = "Gazing"

    if vertical < -3:
        text += " Down"
    elif vertical > 3:
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

            # Extract head orientation angles
            vertical = angles[0] * 360
            horizontal = angles[1] * 360
            # Determine head position
            text = get_direction(horizontal, vertical)

            # Project nose direction onto 2D
            nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + horizontal * 10), int(nose_2d[1] - vertical * 10))

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
