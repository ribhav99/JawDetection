import cv2
import dlib
import numpy as np
from tqdm import tqdm

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/ribhavkapur/Desktop/everything/College/CS/MODELS/shape_predictor_68_face_landmarks.dat")
video_file = "kyleDay1convo_1_iPhone.mov"
out_video_file = video_file.split(".")[0] + "_annotated.mp4"
jaw_values_file = video_file.split(".")[0] + "_values.npy"
jaw_distances_file = video_file.split(".")[0] + "_distances .npy"

def calculate_mouth_open(shape):
    # # Mouth landmarks (48-67)
    # top_lip = shape[50:53] + shape[61:64]
    # low_lip = shape[56:59] + shape[65:68]

    # top_mean = np.mean(top_lip, axis=0)
    # low_mean = np.mean(low_lip, axis=0)

    # return np.linalg.norm(top_mean - low_mean)
    nose_point = 30
    jaw_point = 8
    dist = shape[jaw_point][1] - shape[nose_point][1]
    return dist


# Open a video file or capture device.
video_capture = cv2.VideoCapture(video_file)

# Get video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('annotated_video.mp4', fourcc, fps, (frame_width, frame_height))

frame_skip = 20
frame_num = 0
starting = 10000
jaw_distances = []
frames = []
pbar = tqdm(total=total_frames)
while True:
    # # if frame_num % frame_skip != 0:
    # # frame_num += 1
    # # pbar.update(1)
    # #     continue
    # if starting < frame_num:
    # frame_num += 1
    # pbar.update(1)
    #     continue
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # frames.append(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    
    for face in faces:
        shape = predictor(gray, face)
        shape = [(p.x, p.y) for p in shape.parts()]
        
        mouth_open = calculate_mouth_open(shape)
        
        # Draw the facial landmarks on the frame
        # for i, (x, y) in enumerate(shape):
        #     # cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        #     # cv2.putText(frame, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        #     if i == 30 or i == 8:
        #         cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        #         cv2.putText(frame, f"{str(x), str(y)}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display the mouth open value
        jaw_distances.append(mouth_open)
        # cv2.putText(frame, f'Mouth Open: {mouth_open:.2f}', (face.left(), face.top() - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # # Display the resulting frame
    # cv2.imshow('Video', frame)
    
    # # Press 'q' to exit the video window
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    frame_num += 1
    pbar.update(1)
    
pbar.close()

# Release the video capture object

jaw_distances = np.array(jaw_distances)
min_value = np.min(jaw_distances)
max_value = np.max(jaw_distances)
jaw_values = 10 * (jaw_distances - min_value) / (max_value - min_value)
np.save(jaw_values_file, jaw_values)
np.save(jaw_distances_file, jaw_distances)

# Second pass: annotate and write frames
frame_num = 0
for frame in tqdm(frames):
    jaw_value = jaw_values[frame_num]
    cv2.putText(frame, f'Jaw Value: {jaw_value:.2f}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    out.write(frame)
frame_num += 1

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
