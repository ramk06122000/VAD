# import cv2
# import os

# def save_frames_from_videos(input_folder, output_folder):
#     # Get a list of all video files in the input folder
#     video_files = [f for f in os.listdir(input_folder) if f.endswith('.avi') or f.endswith('.mp4')]

#     # Loop through the video files
#     for video_file in video_files:
#         video_path = os.path.join(input_folder, video_file)
#         video_name = os.path.splitext(video_file)[0]  # Get the video name without extension

#         # Create a folder for this video's frames
#         video_output_folder = os.path.join(output_folder, video_name)
#         os.makedirs(video_output_folder, exist_ok=True)

#         # Open the video file
#         cap = cv2.VideoCapture(video_path)

#         # Get the frames per second (fps) and frame count
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         # Loop through the frames
#         for i in range(frame_count):
#             # Read the frame
#             ret, frame = cap.read()

#             # Save the frame
#             frame_filename = os.path.join(video_output_folder, f'{i:04d}.jpg')
#             cv2.imwrite(frame_filename, frame)

#         # Release the video capture object
#         cap.release()

# # Example usage
# input_folder = 'shanghaitech\\training\\videos'
# output_folder = 'train_frame'

# save_frames_from_videos(input_folder, output_folder)

#=============================================================================================
#load and see npy files
import numpy as np
x= np.load('labels\\frame_labels_ped2.npy')

print("x:",x,"\n")
import pandas as pd
df = pd.DataFrame(x)

# Save the DataFrame as a CSV file
df.to_csv('n.csv', index=False)

#===============================================================================================
# #convert a mat to npy

# import numpy as np
# from scipy.io import loadmat

# # Load the .mat file

# import numpy as np
# from scipy.io import loadmat

# # Load the .mat file
# mat_file = loadmat('ped1\\ped1.mat')

# # Extract the 'gt' variable
# data = mat_file['gt']

# # Find the maximum shape among all sub-arrays
# max_shape = max(cell.shape for cell in data.flat)

# # Pad sub-arrays to the maximum shape
# processed_data = [np.pad(cell, ((0, max_shape[0] - cell.shape[0]), (0, max_shape[1] - cell.shape[1])), mode='constant', constant_values=0) for cell in data.flat]

# # Stack the processed data into a single NumPy array
# stacked_data = np.stack(processed_data)

# # Define the output file name
# output_file = 'output.npy'

# # Save the processed data as a .npy file
# np.save(output_file, stacked_data)

# print(f'{output_file} saved successfully.')
