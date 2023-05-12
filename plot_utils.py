from myutils import get_pred_box,get_image_annotation_pairs
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_detection_sample(model ,img_path,label_path):
    #img_paths ,label_paths = get_image_annotation_pairs(test_dir)
    #img_path = img_paths[num]
    #label_path = label_paths[num]
    # make sure to get the right couple
    print(f"{img_path}\n{label_path}")
    # plot detection
    img2 = get_pred_box(model ,img_path)
    plt.figure(figsize=(8 ,4))
    plt.tight_layout()
    plt.imshow(img2)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
    plt.show()


def add_speed(data, num_particle):
    # Iterate through each droplet
    data = data[data['detected'] == 1].copy()
    for i in range(1, num_particle + 1):
        # Calculate the differences in x and y coordinates
        data[f'dx{i}'] = data[f'x{i}'].diff()
        data[f'dy{i}'] = data[f'y{i}'].diff()

        # Calculate the time difference
        data['dt'] = data['time'].diff()

        # Calculate the x and y velocities
        data[f'dx{i}'] = data[f'dx{i}'] / data['dt']
        data[f'dy{i}'] = data[f'dy{i}'] / data['dt']

        # Calculate the speed
        data[f'speed{i}'] = np.sqrt(data[f'dx{i}'] ** 2 + data[f'dy{i}'] ** 2)

    data = data.fillna(0)
    # Drop the time difference column
    data = data.drop('dt', axis=1)

    return data


def get_dynamics(df, num_particle):
    df = add_speed(df, num_particle)

    xlist = [f"x{i}" for i in range(1, num_particle + 1)]
    dxlist = [f"dx{i}" for i in range(1, num_particle + 1)]

    ylist = [f"y{i}" for i in range(1, num_particle + 1)]
    dylist = [f"dy{i}" for i in range(1, num_particle + 1)]

    slist = [f"speed{i}" for i in range(1, num_particle + 1)]

    # get stuff column-wise
    t = df["time"].to_numpy()
    xc = df[xlist].to_numpy()
    dx = df[dxlist].to_numpy()

    yc = df[ylist].to_numpy()
    dy = df[dylist].to_numpy()

    speed = df[slist].to_numpy()

    return t, xc, yc, dx, dy, speed



def plot_speed(data,num_particle,s=40):
    # speed map
    sns.set(style='ticks', font_scale=1.5)
    _, xc, yc, _, _, speed = get_dynamics(data, num_particle)
    for i in range(num_particle):
        fig,ax = plt.subplots(figsize=(10,8))
        ax.plot(xc[0,i], yc[0,i], 'ks', label='initial point', markersize=15)
        ax.plot(xc[-1,i], yc[-1,i], 'ko', label='terminal point', markersize=15)
        ax = sns.scatterplot(x=xc[:,i], y=yc[:,i], hue=speed[:,i], palette='coolwarm', s = 80)
        ax.set(xlabel=None, ylabel=None)
        norm = plt.Normalize(speed.min(), speed.max())
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        sm.set_array([])
        ax.invert_yaxis()
        # Remove the legend and add a colorbar
        ax.figure.colorbar(sm, label='speed[px/sec]')
        ax.get_legend().remove()
        ax.grid(False)
        # Remove x and y ticks
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()


# Calculate the minimum distance between particles for each detected frame
def min_distance(row, num_particles):
    if row['detected'] == 1:
        coords = [(row[f'x{i+1}'], row[f'y{i+1}']) for i in range(num_particles)]
        distances = [((x1-x2)**2 + (y1-y2)**2)**0.5 for (x1, y1), (x2, y2) in combinations(coords, 2)]
        return min(distances)
    else:
        return float('nan')


def plot_missed_frames(data, num_particles):
    # Calculate the minimum distance between particles for each row
    data["min_distance"] = data.apply(lambda row: min_distance(row, num_particles), axis=1)

    # Get detected and missed frames
    detected = data[data['detected'] == 1]
    missed = data[data['detected'] == 0].copy()

    # Approximate min_distance for missed frames based on the closest detected frame before that frame
    missed["min_distance"] = missed["frame_id"].apply(
        lambda x: detected[detected["frame_id"] < x]["min_distance"].iloc[-1])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(missed["min_distance"], missed["time"], color="red", marker="o", s=10, label="Missing Frames")
    ax.scatter(detected["min_distance"], detected["time"], color="blue", marker="o", s=10, alpha=0.1,
               label="Detected Frames")

    # Add a vertical line for the mean minimum distance
    mean_min_distance = detected["min_distance"].mean()
    ax.axvline(x=mean_min_distance, color="green", linestyle="-", linewidth=3, label="Mean Minimum Distance")

    ax.set_xlabel("Minimum Distance(pixel)", fontsize=15)
    ax.set_ylabel("Time", fontsize=15)
    ax.legend()
    #ax.set_rasterized(True)

    fig.tight_layout()
    #fig.subplots_adjust(bottom=0.1)
    #fig.savefig('figures/missing_frames.eps', format='eps', dpi=300)

    plt.show()


def plot_mAP(data):
    epochs = data['epoch']
    mAP_05 = data['metrics/mAP50(B)']
    mAP_0595 = data['metrics/mAP50-95(B)']
    fig,ax = plt.subplots(figsize=(8,6))
    ax.plot(epochs, mAP_05, label='mAP@0.5', linestyle='--')
    ax.plot(epochs, mAP_0595, label='mAP@0.5:0.95')
    ax.set_xlabel('epochs')
    ax.set_ylabel('score')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_detection_sample(model,test_dir,num):

    img_paths, label_paths = get_image_annotation_pairs(test_dir)
    img_path = img_paths[num]
    label_path = label_paths[num]
    # make sure to get the right couple
    print(f"displaying {img_path}\n{label_path}")
    # plot detection
    img2 = get_pred_box(model, img_path)
    plt.figure(figsize=(10, 6))
    plt.tight_layout()
    plt.imshow(img2)
    plt.gca().set_axis_off()
    plt.show()