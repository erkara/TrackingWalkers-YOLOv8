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
    """
    Get some information about the video source
    :param video_path:
    :return: None
    """
    cap = cv2.VideoCapture(video_path)
    frameRate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_lenght_seconds = total_frames / frameRate
    video_lenght_minutes = video_lenght_seconds / 60
    print(f'frame_rate: {frameRate}\ntotal_frames: {total_frames}')
    print(f'length(s): {video_lenght_seconds:0.1f}\nlength(mins) : {video_lenght_minutes:0.2f}')
    cap.release()


def capture_frames(video_path, image_dir, save_interval=1):
    """
    capture frames in every save_intervals(seconds)
    """
    if os.path.exists((image_dir)):
        shutil.rmtree(image_dir)
        os.makedirs(image_dir)
    else:
        os.makedirs(image_dir)

    cap = cv2.VideoCapture(video_path)
    frameRate = cap.get(5)
    count = 0
    while (cap.isOpened()):
        # get the current frame number
        frameId = cap.get(1)

        # read the frame
        success, frame = cap.read()
        if (success != True):
            break

        # framID/frameRate=0 means every second
        if (frameId % math.floor(frameRate * save_interval) == 0):
            count += 1
            filename = os.path.join(image_dir, str(int(frameId)) + '.jpg')
            cv2.imwrite(filename, frame)
    cap.release()
    print(f'{count} frames captured! and saved to {image_dir}')


def train_valid_test_split(image_dir, dest_dir,
                           train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    """
    This function first creates a folder structure for train/valid/test data as root_dir/X/images
    and root_dir/X/label. We then get images from image_dir and save to these folders accordingly
    based on the provided rations. Run only once, no safeguard to avoid duplicates.
    """

    # dont change this, there is a bug in YOLOv8 that it requires your data to be in a folder
    # named "datasets".
    dest_dir = os.path.join(dest_dir, 'datasets')
    if os.path.exists((dest_dir)):
        shutil.rmtree(dest_dir)
        os.makedirs(dest_dir)
    else:
        os.makedirs(dest_dir)

    # create the folder structure

    folders = ["train", "valid", "test"]
    for folder in folders:
        os.makedirs(f"{dest_dir}/{folder}/images", exist_ok=True)
        os.makedirs(f"{dest_dir}/{folder}/labels", exist_ok=True)

    train_dir = f"{dest_dir}/train/images"
    valid_dir = f"{dest_dir}/valid/images"
    test_dir = f"{dest_dir}/test/images"

    # get all image paths and shuffle them up
    image_paths = []
    for (dirpath, dirnames, filenames) in os.walk(image_dir):
        for names in filenames:
            image_paths.append(os.path.join(dirpath, names))

    random.shuffle(image_paths)

    train_image_list, temp = np.split(np.array(image_paths), [math.ceil(len(image_paths) * (train_ratio))])
    valid_image_list, test_image_list = np.split(temp,
                                                 [math.ceil(len(temp) * valid_ratio / (valid_ratio + test_ratio))])

    for train_image in train_image_list:
        shutil.copy(train_image, train_dir)

    for valid_image in valid_image_list:
        shutil.copy(valid_image, valid_dir)

    for test_image in test_image_list:
        shutil.copy(test_image, test_dir)

    print(f"{len(train_image_list)} training images copied to {train_dir}\
    \n{len(valid_image_list)} validation images copied to {valid_dir}\
    \n{len(test_image_list)} testing images copied to {test_dir}")

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
    root_dir=train_dir,valid dir or test_dir. Beware of uncessary files
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
    Return the image with predicted bounding boxes with confidance
    scores. Notice that this is only for single class prediction
    """

    detection = model(img_path, verbose=False)
    detection = detection[0].boxes
    locs = detection.xyxy.detach().cpu().numpy()
    confidence_scores = detection.conf.detach().cpu().numpy()
    img = cv2.imread(img_path)
    if (detection.cls.sum() == 0) & (detection.shape[0] != 0):
        for i in range(detection.shape[0]):
            xmin = int(locs[i][0])
            ymin = int(locs[i][1])
            xmax = int(locs[i][2])
            ymax = int(locs[i][3])
            img = cv2.rectangle(img, (xmin, ymax), (xmax, ymin), color, 1)
            conf = str(round(confidence_scores[i], 2))
            img = cv2.putText(img, conf, (xmin, ymin - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img


def get_image_box(img_path, label_path, color=(0, 255, 0)):
    """
    Given image and YOLO format label, return the image with bounding box(es)
    Return the box coordinated if necessary
    """
    img = cv2.imread(img_path)
    H, W, _ = img.shape

    # read annotation files in "txt" format
    annot_df = pd.read_csv(label_path, sep=' ', header=None)
    for i in range(annot_df.shape[0]):
        # class_id,x0,y0,w0,h0
        x0, y0, w0, h0 = annot_df.iloc[i].to_list()[1:]

        # class_id = annot_array[0]
        xmax = int((x0 + w0 / 2) * W)
        xmin = int((x0 - w0 / 2) * W)
        ymin = int((y0 + h0 / 2) * H)
        ymax = int((y0 - h0 / 2) * H)

        # pt1:top_left, pt2:bottom_right
        img = cv2.rectangle(img, (xmin, ymax), (xmax, ymin), color, 1)
    #         img = cv2.putText(img, 'original', (xmin, ymin - 5),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img


def get_bb_centers(detection):
    # bounding box coordindates
    detection = detection[0].boxes
    locs = detection.xyxy.detach().cpu().numpy()
    # confidance scores
    confidence_scores = detection.conf.detach().cpu().numpy()
    m = locs.shape[0]
    bb_centers = np.zeros((m, 2))
    for i in range(m):
        xc = (locs[i][0] + locs[i][2]) / 2
        yc = (locs[i][1] + locs[i][3]) / 2
        bb_centers[i] = [xc, yc]

    return bb_centers, confidence_scores


def get_image_with_bounding_box(detection, img, color=(0, 0, 255)):
    detection = detection[0].boxes
    locs = detection.xyxy.detach().cpu().numpy()
    for i in range(detection.shape[0]):
        xmin = int(locs[i][0])
        ymin = int(locs[i][1])
        xmax = int(locs[i][2])
        ymax = int(locs[i][3])
        img = cv2.rectangle(img, (xmin, ymax), (xmax, ymin), color, 1)
    return img


def is_detected(detection, conf_thresold, num_particle):
    detection = detection[0].boxes
    # number of classes detected
    classes = (detection.cls.detach().cpu().numpy() == 0).sum()
    # accept only detections above a certain conf_thresold and count them
    conf_scores = (detection.conf.detach().cpu().numpy() > conf_thresold).sum()

    # 0:no detection, else we strictly accept "nd" classes and confidence scores
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
    cost = [[0 for i in range(num_particle)] for j in range(num_particle)]
    for i in range(num_particle):
        for j in range(num_particle):
            cost[i][j] = np.sum((f1[i] - f2[j]) ** 2)
    row_ind, col_ind = optimize.linear_sum_assignment(cost)
    temp_all = f2[col_ind[np.argsort(row_ind)]].flatten()
    temp_all = np.insert(temp_all, 0, frameID)
    temp_all = np.insert(temp_all, 1, time)
    temp_all = np.insert(temp_all, 2, detected)
    temp_all = np.append(temp_all, conf_score)

    return f2[col_ind[np.argsort(row_ind)]], temp_all


def generate_colors(num_particle):
    # Define the first four colors as green, blue, yellow, and magenta
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    # Generate additional colors randomly
    for _ in range(num_particle - 4):
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    return colors


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


# after revision


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

        detection = model(frame, verbose=False)
        centers, conf_score = get_bb_centers(detection)

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

        else:
            detected = 0
            missed_frame_row = [frame_counter, time_second, int(detected)] + [np.nan] * (len(data.columns) - 3)
            data.loc[frame_counter] = missed_frame_row

        # exclude visualization time from the inference
        simulation_time += time.time() - inference_start

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(
        f'experiment: {save_name} detection_rate: {detect_counter}/{frame_counter} = {100 * (detect_counter / frame_counter):0.3f}%')

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

    print(
        f'experiment: {save_name} detection_rate: {detect_counter}/{frame_counter} = {100 * (detect_counter / frame_counter):0.2f}%')

    csv_path = os.path.join(save_dir, f'{save_name}.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    data.to_csv(csv_path, index=False)
    print(f"results are saved to {csv_path}...")

    return detect_counter, frame_counter, data, simulation_time
