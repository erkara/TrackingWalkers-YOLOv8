import pandas as pd
import numpy as np
import cv2
import math
import os
from scipy import optimize
import shutil
import random
import time
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def GetVideoInfo(video_path):
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


def CaptureFrames(video_path, image_dir, save_interval=1):
    """
    capture frames in every save_intervals(seconds)
    """
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


def TrainValidTestSplit(image_dir, root_dir,
                        train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    """
    This function first creates a folder structure for train/valid/test data as root_dir/X/images
    and root_dir/X/label. We then get images from image_dir and save to these folders accordingly
    based on the provided rations. Run only once, no safeguard to avoid duplicates.
    """
    # create the folder structure
    folders = ["train", "valid", "test"]
    for folder in folders:
        os.makedirs(f"{root_dir}/{folder}/images", exist_ok=True)
        os.makedirs(f"{root_dir}/{folder}/labels", exist_ok=True)

    train_dir = f"{root_dir}/train/images"
    valid_dir = f"{root_dir}/valid/images"
    test_dir = f"{root_dir}/test/images"

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


def GetImageAnnotationPairs(root_dir):
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


def CreateData(source_dir, dest_dir, n=1, exp='_', remove=True, rename=False):
    """Grab n images/labels from source_dir and move them to dest_dir
    accordingly. As usual, source_dir and dest_dir have the same structure.
    If necessary, rename them using 'exp' name. Usually it is
    a good idea to remove all the files in dest_dir.
    """
    img_paths, label_paths = GetImageAnnotationPairs(source_dir)
    # get all if n=-1
    if n == -1:
        n = len(img_paths)
    image2copy = img_paths[:n]
    label2copy = label_paths[:n]
    image_dir = os.path.join(dest_dir, 'images')
    label_dir = os.path.join(dest_dir, 'labels')

    # remove any existing files
    if remove:
        for file in os.scandir(image_dir):
            os.remove(file.path)
        for file in os.scandir(label_dir):
            os.remove(file.path)
    # copy new files
    for image in image2copy:
        shutil.copy(image, image_dir)
        if rename:
            old_name = image.split("/")[-1]
            old_path = os.path.join(image_dir, old_name)
            new_name = f"{exp}_{old_name}"
            new_path = os.path.join(image_dir, new_name)
            os.rename(old_path, new_path)
    for labels in label2copy:
        shutil.copy(labels, label_dir)
        if rename:
            old_name = labels.split("/")[-1]
            old_path = os.path.join(label_dir, old_name)
            new_name = f"{exp}_{old_name}"
            new_path = os.path.join(label_dir, new_name)
            os.rename(old_path, new_path)
    print(f"removed all files from {image_dir} and created {len(image2copy)} images")


def GetPredBboxes(model, img_path, color=(255, 0, 0)):
    """
    Return the image with predicted bounding boxes with confidance
    scores. Notice that this is only for single class prediction
    """
    pred = model(img_path)
    pred = pred.xyxy[0].detach().cpu().numpy()
    img = cv2.imread(img_path)
    if (pred[0][5] == 0) & (pred.size != 0):
        for i in range(pred.shape[0]):
            xmin = int(pred[i][0])
            ymin = int(pred[i][1])
            xmax = int(pred[i][2])
            ymax = int(pred[i][3])
            img = cv2.rectangle(img, (xmin, ymax), (xmax, ymin), color, 1)
            conf = str(round(pred[i][4], 2))
            img = cv2.putText(img, conf, (xmin, ymin - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img


def GetImageBox(img_path, label_path, color=(0, 255, 0)):
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


def GetCenters(detection):
    # (x_min,y_min,x_max,y_max,confidence,class)
    pred = detection.xyxy[0].detach().cpu().numpy()
    centers = np.zeros((pred.shape[0], 2))
    conf_score = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        xc = (pred[i][0] + pred[i][2]) / 2
        yc = (pred[i][1] + pred[i][3]) / 2
        centers[i] = [xc, yc]
        conf_score[i] = pred[i][4]

    return centers, conf_score


def IsDetected(detection, conf_thresold, nd):
    ##(x_min,y_min,x_max,y_max,confidence,class)->ndX6 arrray; actual pixels
    # number of classes detected
    classes = (detection.xyxy[0][:, 5] == 0).sum()
    # accept only detections above a certain conf_thresold and count them
    conf_scores = (detection.xyxy[0][:, 4] > conf_thresold).sum()

    # 0:no detection, else we strictly accepts nd classes and confidance scores
    size = detection.xyxy[0].numel()
    if size != 0:
        if classes == nd & conf_scores == nd:
            return True
    else:
        return False


def AddSpeed2DataFrameMulti(df, nd):
    # find the zero time stamps(except the first one) and replace them with
    # the appropriate time starting from the last nonzero time using dt
    time_zeros = df[(df['time'].index > 1) & (df['time'] == 0.0)]
    if time_zeros.size != 0:
        t1 = df.loc[time_zeros.index[0] - 1, 'time']
        t2 = df.loc[time_zeros.index[0] - 2, 'time']
        dt = t1 - t2
        new_times = [(i + 1) * dt + t1 for i in range(time_zeros.shape[0])]
        df.loc[time_zeros.index, 'time'] = new_times

    t = df['time'].values
    for i in range(nd):
        xc = df.iloc[:, 2 * (i + 1)].values
        yc = df.iloc[:, 2 * (i + 1) + 1].values
        dx = np.zeros(xc.size)
        dx[1:] = np.diff(xc) / np.diff(t)
        dy = np.zeros(yc.size)
        dy[1:] = np.diff(yc) / np.diff(t)
        speed = np.sqrt(dx ** 2 + dy ** 2)
        df[f'dx{i + 1}'] = dx
        df[f'dy{i + 1}'] = dy
        df[f'speed{i + 1}'] = speed
    return df


def GetColNames(nd):
    l = []
    for i in range(nd):
        xname = f'x{i + 1}'
        l.append(xname)
        yname = f'y{i + 1}'
        l.append(yname)
    for i in range(nd):
        c_name = f'c{i + 1}'
        l.append(c_name)

    l.insert(0, 'frame_id')
    l.insert(1, 'time')

    return l


def GetNextCoordHungarian(nd, f1, f2, time, frameID, conf_score):
    cost = [[0 for i in range(nd)] for j in range(nd)]
    for i in range(nd):
        for j in range(nd):
            cost[i][j] = np.sum((f1[i] - f2[j]) ** 2)
    row_ind, col_ind = optimize.linear_sum_assignment(cost)
    temp_all = f2[col_ind[np.argsort(row_ind)]].flatten()
    temp_all = np.insert(temp_all, 0, frameID)
    temp_all = np.insert(temp_all, 1, time)
    temp_all = np.append(temp_all, conf_score)

    return f2[col_ind[np.argsort(row_ind)]], temp_all


def TrackDroplet(model, nd, video_path, conf_thresold, save_dir, save_name, show_trace=False):
    """

    Args:
        model: YOLOV5 model
        nd: number of droplets
        video_path: droplet video source
        conf_thresold: do not accept the detections below this thresold
        save_dir: save the tracking video
        save_name: name of the video and dataframe to be saved
        show_trace: display trace in real time,defaul:false

    Returns:
        data:   Given the experiment with single or multiple droplets, inspect real time tracking and
        save the location and speed information for the individual droplets as a dataframe
        Notice the structure of data.
    """

    if show_trace:
        name = f"{save_name}_trace"
    else:
        name = f"{save_name}_bb"

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h)

    myvideo = cv2.VideoWriter(f'{save_dir}/{name}.avi', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    drop = []
    for i in range(nd):
        drop.append([])

    frame_counter = 0
    detect_counter = 0

    # BGR--> green,red,blue
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255)]

    data = pd.DataFrame(columns=GetColNames(nd))
    while cap.isOpened():
        success, frame = cap.read()
        if (success != True):
            break
        frameID = cap.get(1)

        frame_counter += 1

        time_second = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

        detection = model(frame)
        if IsDetected(detection, conf_thresold, nd):
            # openCV returns 0 time stamp for the few frames at the end, bug in progress.

            detect_counter += 1
            np_image = np.squeeze(detection.render(labels=False))
            centers, conf_score = GetCenters(detection)
            if detect_counter == 1:
                C0 = centers
                for i in range(nd):
                    drop[i].append(C0[i, :])

                row0 = C0.flatten()
                row0 = np.insert(row0, 0, int(frameID))
                row0 = np.insert(row0, 1, time_second)
                row0 = np.append(row0, conf_score)
                data.loc[detect_counter] = row0
                t0 = time_second

            if detect_counter > 1:
                C1, temp_all = GetNextCoordHungarian(nd, C0, centers, time_second, int(frameID), conf_score)
                data.loc[detect_counter] = temp_all
                C0 = C1

                for i in range(nd):
                    drop[i].append(C1[i, :])

                for i in range(nd):
                    pts = np.array(drop[i], np.int32)
                    if show_trace:
                        cv2.polylines(np_image, [pts], False, colors[i], thickness=2)
                        cv2.putText(np_image, f'{i}', (int(C0[i][0]), int(C0[i][1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[i], 2)
                    else:
                        cv2.putText(np_image, f'{i}', (int(C0[i][0]), int(C0[i][1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[i], 2)
                myvideo.write(np_image)
                cv2.imshow(f'{name}', np_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f'{detect_counter}/{frame_counter}={100 * (detect_counter / frame_counter):0.2f}% detected! in \
        {save_name} experiment\n')
    data = AddSpeed2DataFrameMulti(data, nd)
    data.to_csv(f'{save_dir}/{save_name}.csv', index=False)
    print(f"results are saved to {save_dir}...")

    return detect_counter, frame_counter, data


def UpdateData(data_dir, dest_dir, N):
    """
    Copy N images from data_dir/train and data_dir/valid
    to dest_dir/train and dest_dir/valid. We do this to dynamically
    train a model with increasing number of train/valid images.
    """
    K = int(np.ceil((N / 0.7) * 0.2))
    root_train_dir = f"{data_dir}/train"
    root_valid_dir = f"{data_dir}/valid"

    temp_dir = f"{dest_dir}/temp"
    if not os.path.exists(temp_dir):
        shutil.copytree(data_dir, temp_dir)

    temp_train_imagedir = os.path.join(temp_dir, 'train/images')
    temp_train_labeldir = os.path.join(temp_dir, 'train/labels')

    temp_valid_imagedir = os.path.join(temp_dir, 'valid/images')
    temp_valid_labeldir = os.path.join(temp_dir, 'valid/labels')

    x = [root_train_dir, root_valid_dir]
    y = [temp_train_imagedir, temp_valid_imagedir]
    z = [temp_train_labeldir, temp_valid_labeldir]

    i = 0
    for (root_dir, temp_image_dir, temp_label_dir) in zip(x, y, z):
        img_paths, lab_paths = GetImageAnnotationPairs(root_dir)
        if i == 1:
            N = K
        if N > len(img_paths):
            print(f"{root_dir} does not have {N} images, switching to N={len(img_paths) - 1}!\n")
            N = len(img_paths) - 1

        img2copy = img_paths[:N]
        lab2copy = lab_paths[:N]

        # empty temp_train_dir then populate it
        if len(os.listdir(temp_image_dir)) != 0:
            for file in os.scandir(temp_image_dir):
                os.remove(file.path)
            for file in os.scandir(temp_label_dir):
                os.remove(file.path)
            for image in img2copy:
                shutil.copy(image, temp_image_dir)
            for labels in lab2copy:
                shutil.copy(labels, temp_label_dir)
        else:
            for image in img2copy:
                shutil.copy(image, temp_image_dir)
            for labels in lab2copy:
                shutil.copy(labels, temp_label_dir)
        print(f"updated {temp_image_dir} folder with {N} new images/labels out of {len(img_paths)} from {root_dir}")

        i += 1


def OptimumTrainImages(data_dir, project_dir, max_image_number=150, start_image_num=5, final_image_num=150,
                       num_interval=5, epoch=50, save_name='test_scores'):
    """
    Inspect optimum number of images to train your model. Every action is performed in project_dir/temp folder
    then we remove it. We train/test the model for number of images between from start_image_num and final_image_num
    with num_interval intervals for epoch numbers.
    """

    if final_image_num > max_image_number:
        print(f"dont have {final_image_num} images, switching to {max_image_number} images")
        final_image_num = max_image_number
    if start_image_num < 4:
        print("cannot keep 0.7/0.2 train/test ratio with {start_image_num} intial images switching to 4")
        start_image_num = 4

    sample_numbers = [num for num in range(start_image_num, final_image_num + 1) if num % num_interval == 0]

    print(f"we will train model with {sample_numbers} training images")

    # save number of train_images, 'mAP@0.5', 'mAP@0.5..0.95' to a dataframe in the current directory
    pd.DataFrame(columns=['num_train_image', 'mAP@0.5', 'mAP@0.5..0.95']).to_csv(f"{project_dir}/{save_name}.csv",
                                                                                 index=False)

    # yaml file /yolov5/custom_data/temp_train.yml must point temp_train folder
    data = "yolov5/custom_data/temp.yml"


    project_name = "temp_results"
    # the best model is default saved here, and we train yolov5s.
    model = f"{project_dir}/{project_name}/weights/best.pt"
    optimizer = "Adam"

    for sample_num in sample_numbers:
        # update the number of images in temp_train folder
        UpdateData(data_dir, project_dir, sample_num)
        # train the model with sample_num images
        print(f"beginning traning with {sample_num} images")
        time.sleep(5)
        os.system(f"python yolov5/train.py --data {data} --weights yolov5/yolov5s.pt \
                  --epoch {epoch} --optimizer {optimizer}  \
                  --project {project_dir} --name {project_name} \
                  --cache --exist-ok --noval --seed 0")

        # wait for the model to be saved. Then start testing
        time.sleep(5)
        print(f"testing the best model trained with {sample_num} images")
        os.system(f"python yolov5/val_erdi.py --data {data} --task test --weights {model} --num_train {sample_num}\
        --mysave_dir {project_dir} --mycsv_name {save_name}")

    # remove the files, we don't need them
    if os.path.exists(f"{project_dir}/{project_name}"):
        shutil.rmtree(f"{project_dir}/{project_name}")
        print(f"removed {project_dir}/{project_name}")
    if os.path.exists(f"{project_dir}/temp"):
        shutil.rmtree(f"{project_dir}/temp")
        print(f"{project_dir}/temp")

    # plot the results
    df = pd.read_csv(f"{project_dir}/{save_name}.csv")
    num_train = df['num_train_image']
    mAP_05 = df['mAP@0.5']
    mAP_0595 = df['mAP@0.5..0.95']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(num_train, mAP_05, label='mAP@0.5', marker='o')
    ax.plot(num_train, mAP_0595, label='mAP@0.5:0.95', linestyle='--', marker='o')
    ax.set_xlabel('#training images')
    ax.set_ylabel('score')
    ax.grid(True)
    ax.legend()
    plt.show()


def TrackMultipleExperiments(exp_dict, video_root_dir, model, conf_thresold=0.45,
                             save_dir="./", name="fdr", show_trace=False):
    """
    Assuming all videos has mp4 extension and located in video_root_dir.
    exp_name is always same with it corresponding video.
    :param exp_dict:{exp_name : #of objects}
    """

    # save frame detection rates in a dataframe
    fdr = pd.DataFrame(columns=['exp_name', 'detected', 'total_frame', 'frame_detection_rate'])
    for index, exp_name in enumerate(exp_dict.keys()):
        nd = exp_dict[exp_name]
        print([exp_name, nd])
        video_path = f"{video_root_dir}/{exp_name}.mp4"
        detected, total_frame, _ = TrackDroplet(model=model, conf_thresold=conf_thresold, nd=nd,
                                                video_path=video_path, save_dir=save_dir,
                                                save_name=exp_name, show_trace=show_trace)
        fdr.loc[index] = [exp_name, detected, total_frame, round(detected / total_frame, 5)]

    fdr.to_csv(f"{save_dir}/{name}.csv", index=False)
