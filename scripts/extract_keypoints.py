import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from multiprocessing import Pool, cpu_count


# Function to extract hand keypoints
def extract_keypoints_worker(args):
    """Worker function for parallel processing"""
    image, label = args
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            # Flatten the 21 keypoints (x, y, z) into a single list
            keypoints = (
                [round(landmark.x, 4) for landmark in hand_landmarks.landmark]
                + [round(landmark.y, 4) for landmark in hand_landmarks.landmark]
                + [round(landmark.z, 4) for landmark in hand_landmarks.landmark]
            )
            return keypoints + [label]
        return None


def process_label_images(dataset_path, label, output_file):
    """Process images of a single label and save to a file incrementally"""
    label_path = os.path.join(dataset_path, label)
    image_label_pairs = []

    # Load images for the current label
    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        image = cv2.imread(img_path)
        if image is not None:
            image_label_pairs.append((image, label))

    # Process the images in parallel to extract keypoints
    data = []
    print(f"Processing {len(image_label_pairs)} images for label '{label}'...")

    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(
            pool.imap_unordered(extract_keypoints_worker, image_label_pairs),
            total=len(image_label_pairs),
        ):
            if result is not None:
                data.append(result)

    # Convert list to numpy array for efficiency
    data_np = np.array(data)

    # Define the schema (column names)
    columns = (
        [f"x{i}" for i in range(21)]
        + [f"y{i}" for i in range(21)]
        + [f"z{i}" for i in range(21)]
        + ["label"]
    )

    # Create a DataFrame using the numpy array and columns
    df = pd.DataFrame(data_np, columns=columns)

    # Check if the file already exists
    if os.path.exists(output_file):
        # If the file exists, append the new data
        existing_df = pd.read_csv(output_file)

        # Append the new DataFrame to the existing one
        df = pd.concat([existing_df, df], ignore_index=True)
        df.to_csv(output_file, index=False)  # Write back the combined DataFrame
    else:
        # If the file doesn't exist, just write the new data
        df.to_csv(output_file, index=False)


# Initialize Mediapipe Hands globally
mp_hands = mp.solutions.hands

# Define dataset path and output file
dataset_path = "asl_alphabet_train/asl_alphabet_train"
output_file = "hand_keypoints.csv"

# Process each label one by one
labels = os.listdir(dataset_path)

# Initialize the output file with column headers if it doesn't exist
columns = (
    [f"x{i}" for i in range(21)]
    + [f"y{i}" for i in range(21)]
    + [f"z{i}" for i in range(21)]
    + ["label"]
)

# Create an empty CSV file with column headers if it doesn't exist
if not os.path.exists(output_file):
    # Create an empty DataFrame with column headers
    empty_df = pd.DataFrame(columns=columns)
    empty_df.to_csv(output_file, index=False)

# Loop over each label and process
for label in labels:
    process_label_images(dataset_path, label, output_file)

print(f"Keypoints extraction completed. Data saved to {output_file}")
