import pandas as pd
import numpy as np
import cv2
import math
import os
import re
from scipy import optimize
import shutil
import random
import time
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')



def get_video_info(video_path):
    # Create a VideoCapture object. 
    cap = cv2.VideoCapture(video_path)
    
    # Get the frame rate of the video. 
    frameRate = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Get the total number of frames in the video.
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # Calculate the total length of the video in seconds.
    video_lenght_seconds = total_frames / frameRate
    
    # Calculate the total length of the video in minutes.
    video_lenght_minutes = video_lenght_seconds / 60
    
    # Print the frame rate and total frames of the video.
    print(f'frame_rate: {frameRate}\ntotal_frames: {total_frames}')
    
    # Print the length of the video in seconds and minutes.
    print(f'length(s): {video_lenght_seconds:0.1f}\nlength(mins) : {video_lenght_minutes:0.2f}')
    
    # Release the VideoCapture object. 
    cap.release()


def capture_frames(video_path, image_dir, save_interval=1):
    """
    Function to capture and save frames from a video file.

    Parameters:
    video_path (str): The path to the video file.
    image_dir (str): The directory to save the images in.
    save_interval (int, optional): The interval (in seconds) between frames to save. Default is 1.
    """
    # Check if the image directory exists
    if os.path.exists((image_dir)):
        # If it does, remove it and recreate it
        shutil.rmtree(image_dir)
        os.makedirs(image_dir)
    else:
        # If it doesn't, create it
        os.makedirs(image_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    frameRate = cap.get(5)

    count = 0
    # Loop through the video frames
    while (cap.isOpened()):
        # get the current frame number
        frameId = cap.get(1)

        # Read the frame
        success, frame = cap.read()

        # If the frame was not successfully read, break from the loop
        if (success != True):
            break

        # Check if the current frame is at an interval of save_interval seconds
        if (frameId % math.floor(frameRate * save_interval) == 0):
            # If it is, increment the count
            count += 1
            # Create the filename for the image
            filename = os.path.join(image_dir, str(int(frameId)) + '.jpg')
            # Write the image to the file
            cv2.imwrite(filename, frame)

    # Release the video file
    cap.release()

    # Print the number of frames captured
    print(f'{count} frames captured! and saved to {image_dir}')


def train_valid_test_split(image_dir, dest_dir,
                           train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    """
    This function first creates a folder structure for train/valid/test data as dest_dir/datasets/X/images
    and dest_dir/datasets/X/label. We then get images from image_dir and save to these folders accordingly
    based on the provided rations.
    """
    # Fix directory path due to known issue in YOLOv8
    dest_dir = os.path.join(dest_dir, 'datasets')

    # If the destination directory already exists, remove and recreate it
    if os.path.exists((dest_dir)):
        shutil.rmtree(dest_dir)
        os.makedirs(dest_dir)
    # If not, simply create it
    else:
        os.makedirs(dest_dir)

    # create the folder structure for train, valid and test directories
    folders = ["train", "valid", "test"]
    for folder in folders:
        os.makedirs(f"{dest_dir}/{folder}/images", exist_ok=True)
        os.makedirs(f"{dest_dir}/{folder}/labels", exist_ok=True)

    # Define specific directories for training, validation and testing images
    train_dir = f"{dest_dir}/train/images"
    valid_dir = f"{dest_dir}/valid/images"
    test_dir = f"{dest_dir}/test/images"

    # Get all image paths from the provided image directory
    image_paths = []
    for (dirpath, dirnames, filenames) in os.walk(image_dir):
        for names in filenames:
            image_paths.append(os.path.join(dirpath, names))

    # Randomly shuffle the image paths
    random.shuffle(image_paths)

    # Split the image paths into train, valid and test lists based on the provided ratios
    train_image_list, temp = np.split(np.array(image_paths), [math.ceil(len(image_paths) * (train_ratio))])
    valid_image_list, test_image_list = np.split(temp,
                                                 [math.ceil(len(temp) * valid_ratio / (valid_ratio + test_ratio))])

    # Copy the images to the respective directories
    for train_image in train_image_list:
        shutil.copy(train_image, train_dir)

    for valid_image in valid_image_list:
        shutil.copy(valid_image, valid_dir)

    for test_image in test_image_list:
        shutil.copy(test_image, test_dir)

    # Print the number of images copied to each directory
    print(f"{len(train_image_list)} training images copied to {train_dir}\
    \n{len(valid_image_list)} validation images copied to {valid_dir}\
    \n{len(test_image_list)} testing images copied to {test_dir}")

    # Create yaml files in the destination directory
    create_yaml_files(dest_dir)


def create_yaml_files(dest_dir):
    # create a yaml file in order YOLOv8 to access our data.
    content = f"""\
        path: {os.path.abspath(dest_dir)}/
        train: train/images  # train images (relative to 'dest_dir') 
        val: valid/images  # val images (relative to 'dest_dir')
        test: test/images # test images (optional)

        nc: 1  # assuming one particle entity
        names: ['droplet']  # class names
        """

    # Create the "sample.yaml" file
    with open(f"{dest_dir}/sample.yaml", "w") as file:
        file.write(content)

    # dummy yaml file to test our model
    content = f"""\
            #this is just a dummy file to test our model by swapping valid-->test
            path: {os.path.abspath(dest_dir)}/
            train: train/images  # test images 
            val: test/images  # val images

            nc: 1  # assuming one particle entity
            names: ['droplet']  # 
            """
    # Create the "sample.yaml" file
    with open(f"{dest_dir}/dummy_test.yaml", "w") as file:
        file.write(content)

    print(f"created sample.yaml and dummy_test.yaml in {dest_dir} ")


def get_image_annotation_pairs(root_dir):
    """
    Get the image-label directory list in an aligned fashion. 
    root_dir=train_dir,valid dir or test_dir. Beware of unnecessary files
    in these directories. No exception handling, I am lazy today...
    """
    img_paths = []
    lab_paths = []
    for (dirpath, dirnames, filenames) in os.walk(root_dir):
        for names in filenames:
            if names.endswith('.jpg'):
                img_paths.append(os.path.join(dirpath, names))
            if names.endswith('.txt'):
                lab_paths.append(os.path.join(dirpath, names))
    img_paths.sort()
    lab_paths.sort()

    return img_paths, lab_paths


def get_pred_box(model, img_path, color=(255, 0, 0)):
    """
    Function to overlay predicted bounding boxes and their confidence scores on an input image.

    Args:
        model (Object): The trained YOLOv8 model used for making predictions.
        img_path (str): The path to the image on which predictions are to be made.
        color (tuple, optional): The color of the bounding boxes and confidence scores. Defaults to red (255, 0, 0).

    Returns:
        img: Image with bounding boxes and confidence scores overlayed.
    """

    # Perform detection on the input image
    detection = model(img_path, verbose=False)
    # Get bounding box object
    detection = detection[0].boxes
    # Detach the bounding box coordinates and convert to numpy
    locs = detection.xyxy.detach().cpu().numpy()
    # Detach the confidence scores and convert to numpy
    confidence_scores = detection.conf.detach().cpu().numpy()
    # Read the image using OpenCV
    img = cv2.imread(img_path)
    # Check if the detected objects belong to the class of interest (class id = 0)
    # and if there's at least one detection
    if (detection.cls.sum() == 0) & (detection.shape[0] != 0):
        # Loop over each detected object
        for i in range(detection.shape[0]):
            # Get bounding box coordinates for the object
            xmin = int(locs[i][0])
            ymin = int(locs[i][1])
            xmax = int(locs[i][2])
            ymax = int(locs[i][3])
            # Draw the bounding box on the image
            img = cv2.rectangle(img, (xmin, ymax), (xmax, ymin), color, 1)
            # Get the confidence score for the object
            conf = str(round(confidence_scores[i], 2))
            # Overlay the confidence score on the image, slightly above the bounding box
            img = cv2.putText(img, conf, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Return the image with overlayed bounding boxes and confidence scores
    return img


def get_image_box(img_path, label_path, color=(0, 255, 0)):
    """
    Given image and YOLO format label, return the image with bounding box(es)
    Return the box coordinated if necessary
    """

    # Read the image file using OpenCV
    img = cv2.imread(img_path)
    # Get the height, width and number of color channels of the image
    H, W, _ = img.shape

    # Read the annotations file (in txt format). Annotations are assumed to be in YOLO format.
    # YOLO format: class_id x_center y_center width height (all normalized to [0,1])
    annot_df = pd.read_csv(label_path, sep=' ', header=None)

    # Loop over all rows (i.e., all annotated objects) in the annotations file
    for i in range(annot_df.shape[0]):
        # Extract the bounding box information. Note that in YOLO format, the
        # bounding box is represented by the center (x0, y0) and width and height (w0, h0).
        x0, y0, w0, h0 = annot_df.iloc[i].to_list()[1:]

        # Convert the normalized coordinates back to pixel coordinates
        # Note: In YOLO, x0, y0, w0, h0 are all normalized to [0, 1] by the image width/height.
        xmax = int((x0 + w0 / 2) * W)  # xmax = (center_x + width / 2) * image_width
        xmin = int((x0 - w0 / 2) * W)  # xmin = (center_x - width / 2) * image_width
        ymin = int((y0 + h0 / 2) * H)  # ymin = (center_y + height / 2) * image_height
        ymax = int((y0 - h0 / 2) * H)  # ymax = (center_y - height / 2) * image_height

        # Draw the bounding box on the image using OpenCV. Here (xmin, ymax) is the top-left point
        # and (xmax, ymin) is the bottom-right point of the box. We're using the provided color
        # and a line thickness of 1 pixel.
        img = cv2.rectangle(img, (xmin, ymax), (xmax, ymin), color, 1)

    # Return the image with the bounding boxes drawn on it
    return img


def get_bb_centers(detection):
    # detection = yolov8_model(img)
    # detection[0].boxes is the object we return bb locations, conf scores etc.
    detection = detection[0].boxes

    # Convert the bounding box coordinates from PyTorch tensor to numpy array
    # and detach it from the computation graph
    locs = detection.xyxy.detach().cpu().numpy()

    # Get the confidence scores for each detected object and
    # convert it from PyTorch tensor to numpy array
    confidence_scores = detection.conf.detach().cpu().numpy()

    # Get the number of detections
    m = locs.shape[0]

    # Initialize a numpy array to store the center coordinates of each bounding box
    bb_centers = np.zeros((m, 2))

    # Calculate the center coordinates for each bounding box
    for i in range(m):
        # Calculate the x-coordinate of the center by averaging the xmin and xmax
        xc = (locs[i][0] + locs[i][2]) / 2
        # Calculate the y-coordinate of the center by averaging the ymin and ymax
        yc = (locs[i][1] + locs[i][3]) / 2
        # Store the center coordinates in the bb_centers array
        bb_centers[i] = [xc, yc]

    # Return the center coordinates and confidence scores of each detected object
    return bb_centers, confidence_scores


def get_image_with_bounding_box(detection, img, color=(0, 0, 255)):
    # detection = yolov8_model(img)
    detection = detection[0].boxes

    # Convert the detection to a numpy array and move to CPU for further processing
    locs = detection.xyxy.detach().cpu().numpy()

    # Iterate over all detected objects
    for i in range(detection.shape[0]):
        # Get the coordinates for the bounding box of the i-th detected object
        # Convert the coordinates to integers for pixel-level precision
        xmin = int(locs[i][0])
        ymin = int(locs[i][1])
        xmax = int(locs[i][2])
        ymax = int(locs[i][3])

        # Draw a rectangle on the image at the detected object's location
        # The color and thickness of the rectangle are set by 'color' and '1' respectively
        img = cv2.rectangle(img, (xmin, ymax), (xmax, ymin), color, 1)

    # Return the image with all detected objects enclosed in rectangles
    return img


def is_detected(detection, conf_thresold, num_particle):
    detection = detection[0].boxes
    # number of classes detected
    classes = (detection.cls.detach().cpu().numpy() == 0).sum()
    # accept only detections above a certain conf_thresold and count them
    conf_scores = (detection.conf.detach().cpu().numpy() > conf_thresold).sum()

    #we strictly accept "num_particle" classes each having confidence scores greater than
    #conf_thresold. This should eliminate false-positives
    size = detection.shape[0]
    if size != 0:
        if classes == num_particle & conf_scores == num_particle:
            return True
    else:
        return False


def get_col_names(num_particle):
    l = []
    for i in range(num_particle):
        xname = f'x{i + 1}'
        l.append(xname)
        yname = f'y{i + 1}'
        l.append(yname)
    for i in range(num_particle):
        c_name = f'c{i + 1}'
        l.append(c_name)

    l.insert(0, 'frame_id')
    l.insert(1, 'time')
    l.insert(2, 'detected')

    return l


def get_next_coord_hungarian(num_particle, f1, f2, time, frameID, conf_score, detected):
    # Initialize a 2D cost matrix with dimensions num_particle x num_particle
    cost = [[0 for i in range(num_particle)] for j in range(num_particle)]

    # Compute the squared Euclidean distance between points in f1 and f2 and store it in the cost matrix
    for i in range(num_particle):
        for j in range(num_particle):
            cost[i][j] = np.sum((f1[i] - f2[j]) ** 2)

    # Use the Hungarian method (also known as the Kuhn-Munkres algorithm or linear sum assignment)
    # to solve the assignment problem and get the optimal assignment of particles in f1 to f2
    row_ind, col_ind = optimize.linear_sum_assignment(cost)

    # Sort f2 based on the assignment and flatten the array
    temp_all = f2[col_ind[np.argsort(row_ind)]].flatten()

    # Insert frameID, time, and detected at the beginning of the sorted and flattened array
    temp_all = np.insert(temp_all, 0, frameID)
    temp_all = np.insert(temp_all, 1, time)
    temp_all = np.insert(temp_all, 2, detected)

    # Append the confidence score at the end of the array
    temp_all = np.append(temp_all, conf_score)

    # Return the sorted f2 and the final array
    return f2[col_ind[np.argsort(row_ind)]], temp_all


def generate_colors(num_particle):
    # Define the first four colors as green, blue, yellow, and magenta
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    # Generate additional colors randomly
    for _ in range(num_particle - 4):
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    return colors




def track_droplet(model, num_particle, video_path, conf_thresold, save_dir, save_name, show_trace=False):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    else:
        os.makedirs(save_dir)

    if show_trace:
        name = f"{save_name}_trace"
    else:
        name = f"{save_name}_bb"

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h)

    myvideo = cv2.VideoWriter(f'{save_dir}/{name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    drop = []
    for i in range(num_particle):
        drop.append([])

    frame_counter = 0
    # openCV returns 0 time stamp for the few frames at the end, bug in progress.
    # Thus, we exclude those from our analysis but keep an eye one them anyway
    broken_frame_counter = 0
    detect_counter = 0

    # get colors
    colors = generate_colors(num_particle)

    #create the dataframe to register all tracking info
    data = pd.DataFrame(columns=get_col_names(num_particle))

    simulation_time = 0
    while cap.isOpened() and frame_counter + broken_frame_counter < total_frames:
        inference_start = time.time()
        success, frame = cap.read()
        if (success != True):
            broken_frame_counter += 1
            continue

        frame_counter += 1
        frameID = cap.get(1)
        time_second = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

        #get detection centers, no filtering
        detection = model(frame, verbose=False)
        centers, conf_score = get_bb_centers(detection)

        #if the detected frame passes our criteria
        if is_detected(detection, conf_thresold, num_particle):
            detected = 1
            detect_counter += 1
            # np_image = detection[0].plot(labels=False)--> bug in the current yolov8 version
            image_with_bb = get_image_with_bounding_box(detection, frame)
            if detect_counter == 1:
                C0 = centers
                for i in range(num_particle):
                    drop[i].append(C0[i, :])

                row0 = C0.flatten()
                row0 = np.insert(row0, 0, int(frameID))
                row0 = np.insert(row0, 1, time_second)
                row0 = np.insert(row0, 2, int(detected))  # Detected
                row0 = np.append(row0, conf_score)

                data.loc[frame_counter] = row0

            if detect_counter > 1:
                C1, temp_all = get_next_coord_hungarian(num_particle, C0, centers, time_second, int(frameID),
                                                        conf_score, int(detected))
                data.loc[frame_counter] = temp_all
                C0 = C1



                for i in range(num_particle):
                    drop[i].append(C1[i, :])
                for i in range(num_particle):
                    pts = np.array(drop[i], np.int32)
                    if show_trace:
                        cv2.polylines(image_with_bb, [pts], False, colors[i], thickness=2)
                        cv2.putText(image_with_bb, f'{i}', (int(C0[i][0]) + 10, int(C0[i][1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[i], 2)
                    else:
                        cv2.putText(image_with_bb, f'{i}', (int(C0[i][0]) + 10, int(C0[i][1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[i], 2)
                myvideo.write(image_with_bb)
                cv2.imshow(f'{name}', image_with_bb)

        #if criteria not met, only register frame_counter,time_second for this frame
        else:
            detected = 0
            missed_frame_row = [frame_counter, time_second, int(detected)] + [np.nan] * (len(data.columns) - 3)
            data.loc[frame_counter] = missed_frame_row

        #total simulation time
        simulation_time += time.time() - inference_start

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f'experiment: {save_name} detection_rate: {detect_counter}/{frame_counter} = {100 * (detect_counter / frame_counter):0.3f}%')

    #save trajectories
    csv_path = os.path.join(save_dir, f'{save_name}.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    data.to_csv(csv_path, index=False)
    print(f"trajectories saved to {csv_path}")

    return detect_counter, frame_counter, simulation_time


# use this function if you are sure there is no ID swithces(maybe single droplet exps)
# since this function is just to measure inference time.
def save_trajectory_only(model, num_particle, video_path, conf_thresold, save_dir, save_name):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_counter = 0
    broken_frame_counter = 0
    detect_counter = 0

    data = pd.DataFrame(columns=get_col_names(num_particle))
    start_time = time.time()
    with tqdm(total=total_frames, desc="Processing frames", ncols=100) as progress_bar:
        while cap.isOpened() and frame_counter + broken_frame_counter < total_frames:
            success, frame = cap.read()
            if (success != True):
                print(f'broken frame')
                broken_frame_counter += 1
                continue
            frame_counter += 1
            frameID = cap.get(1)
            time_second = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

            detection = model(frame, verbose=False)
            centers, conf_score = get_bb_centers(detection)
            if is_detected(detection, conf_thresold, num_particle):
                detected = 1

                detect_counter += 1
                if detect_counter == 1:
                    C0 = centers
                    row0 = C0.flatten()
                    row0 = np.insert(row0, 0, int(frameID))
                    row0 = np.insert(row0, 1, time_second)
                    row0 = np.insert(row0, 2, detected)  # Detected
                    row0 = np.append(row0, conf_score)

                    data.loc[frame_counter] = row0
                if detect_counter > 1:
                    C1, temp_all = get_next_coord_hungarian(num_particle, C0, centers, time_second, int(frameID),
                                                            conf_score, detected)
                    data.loc[frame_counter] = temp_all
                    C0 = C1
            else:
                detected = 0
                missed_frame_row = [frame_counter, time_second, detected] + [np.nan] * (len(data.columns) - 3)
                data.loc[frame_counter] = missed_frame_row

            progress_bar.update(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    simulation_time = time.time() - start_time

    print(f'experiment: {save_name} detection_rate: {detect_counter}/{frame_counter} = {100 * (detect_counter / frame_counter):0.2f}%')

    csv_path = os.path.join(save_dir, f'{save_name}.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    data.to_csv(csv_path, index=False)
    print(f"results are saved to {csv_path}...")

    return detect_counter, frame_counter, data, simulation_time


def track_multiple_experiment(exp_dict, video_root_dir, model, conf_thresold=0.45,
                              save_dir="./", name="fdr", show_trace=False):
    """
    Assuming all videos has mp4 extension and located in video_root_dir.
    exp_name is always same with it corresponding video.
    :param exp_dict:{exp_name : #of objects}
    """

    # save frame detection rates in a dataframe
    fdr = pd.DataFrame(columns=['exp_name', 'detected', 'total_frame', 'frame_detection_rate', 'simulation_time'])
    for index, exp_name in enumerate(exp_dict.keys()):
        num_particle = exp_dict[exp_name]
        print([exp_name, num_particle])
        video_path = f"{video_root_dir}/{exp_name}.mp4"
        detect_counter, total_frame, simulation_time = track_droplet(model, num_particle, video_path,
                                                                     conf_thresold, save_dir,
                                                                     save_name=exp_name, show_trace=show_trace)
        fdr.loc[index] = [exp_name, detect_counter, total_frame, round(detect_counter / total_frame, 5),
                          round(simulation_time, 2)]

    csv_path = f"{save_dir}/{name}.csv"
    fdr.to_csv(csv_path, index=False)
    print(f"frame detection rates saved to {csv_path}")
