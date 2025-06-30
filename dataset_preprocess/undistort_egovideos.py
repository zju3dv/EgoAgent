import os
from projectaria_tools.core import data_provider, calibration, image
import cv2
import matplotlib.pyplot as plt
from PIL import Image 
import json
from tqdm import tqdm

from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.calibration import (
    CameraCalibration,
    distort_by_calibration,
)
import numpy as np
from PIL import Image
from multiprocessing import Pool
import argparse

def get_camera_calibration(
        stream_id, vrs_data_provider
    ) :
    device_calibration = vrs_data_provider.get_device_calibration()
    
    stream_label = vrs_data_provider.get_label_from_stream_id(stream_id)
    camera_calibration = device_calibration.get_camera_calib(stream_label)
    return camera_calibration

def pil_grid(images, max_horiz=np.iinfo(int).max):
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid

def undistort_video_e2e(aria_video_path, image_size, src_calib, dst_calib):
    # read one frame
    # cap = cv2.VideoCapture(aria_video_path)
    # ret, frame = cap.read()
    # frame_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # cap.release()

    # read video
    cap = cv2.VideoCapture(aria_video_path)
    frame_array = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_array.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    undistarted_frame_array = []
    for i in tqdm(range(len(frame_array))):
        frame = frame_array[i]
        # Resize the image to the desired resolution
        # frame = cv2.resize(frame, (448, 448))
        # Undistort the image
        rectified_frame = calibration.distort_by_calibration(frame, dst_calib, src_calib, InterpolationMethod.BILINEAR)
        undistarted_frame_array.append(rectified_frame)

    return frame_array, undistarted_frame_array

def parse_from_vrs(vrs_path, res):
    # Load VRS
    provider = data_provider.create_vrs_data_provider(vrs_path)
    stream_id = provider.get_stream_id_from_label("camera-rgb")

    # Get camera parameters
    camera_calibration = get_camera_calibration(stream_id, provider)
    image_size = camera_calibration.get_image_size()

    # Get source and target calibration
    device_calib = provider.get_device_calibration()
    vrs_calib = device_calib.get_camera_calib('camera-rgb')
    src_calib = vrs_calib.rescale([448, 448], 448/image_size[0])
    dst_focal = src_calib.get_focal_lengths() * res/448
    dst_calib = calibration.get_linear_camera_calibration(res, res, dst_focal[0], "camera-rgb")

    return image_size, src_calib, dst_calib

def undistort_dataset_test():
    # Get video path
    mp4_root = "data/egoexo4d_v2/takes/"
    mp4_take_uids = os.listdir(mp4_root)
    mp4_take_uids.sort()
    print("Number of takes from mp4: ", len(mp4_take_uids))
    postfix_path = "frame_aligned_videos/downscaled/448/aria01_214-1.mp4"
    video_path = os.path.join(mp4_root, mp4_take_uids[0], postfix_path)
    print("Video take uid: ", mp4_take_uids[0])

    # Get VRS path
    vrs_root = "data/egoexo4d_v2/takes/"
    vrs_take_uids = os.listdir(vrs_root)
    vrs_take_uids.sort()
    print("Number of takes from vrs: ", len(vrs_take_uids))
    vrs_path = os.path.join(vrs_root, vrs_take_uids[0], "aria01_noimagestreams.vrs")
    print("VRS take uid: ", vrs_take_uids[0])
    
    # Undistort video
    image_size, src_calib, dst_calib = parse_from_vrs(vrs_path, 448)
    original_frame_array, undistorted_frame_array = undistort_video_e2e(video_path, image_size, src_calib, dst_calib)

    processed_v1_root = "data/egoexo4d_v1/processed_data"
    processed_v1_uids = os.listdir(processed_v1_root)
    processed_v1_uids.sort()
    print("Number of takes from processed v1: ", len(processed_v1_uids))
    print("Processed V1 take uid: ", processed_v1_uids[0])
    processed_image_path = os.path.join(processed_v1_root, processed_v1_uids[0], "aria01_214-1/frame_0.jpg")
    processed_image = Image.open(processed_image_path).resize((448,448))
    print(processed_image.size)

    return original_frame_array, undistorted_frame_array, processed_image

def save_frame_array(frame_array, root_path, take_uid):
    # make directory
    dir_path = os.path.join(root_path, take_uid, "aria01_214-1")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # only save the fisrt frame of every 5 frames
    for i in range(0, len(frame_array), 5):
        frame_path = os.path.join(dir_path, f"frame_{i}.jpg")
        # to rgb
        frame = cv2.cvtColor(frame_array[i], cv2.COLOR_BGR2RGB)
        cv2.imwrite(frame_path, frame)

def undistort_dataset_single_process():
    take_uid_path = "intersection_take_uids.txt"
    with open(take_uid_path, "r") as f:
        take_uids = f.readlines()
    take_uids = [uid.strip() for uid in take_uids]
    print("Number of takes to process: ", len(take_uids))

    # video path
    mp4_root = "data/egoexo4d_v2/takes/"
    postfix_path = "frame_aligned_videos/downscaled/448/aria01_214-1.mp4"
    # VRS path
    vrs_root = "data/egoexo4d_v2/"
    # save image path
    image_root = "data/egoexo4d_v2/processed_data/"

    for take_uid in tqdm(take_uids):
        video_path = os.path.join(mp4_root, take_uid, postfix_path)
        vrs_path = os.path.join(vrs_root, take_uid, "aria01_noimagestreams.vrs")
        image_size, src_calib, dst_calib = parse_from_vrs(vrs_path, 448)
        _, undistorted_frame_array = undistort_video_e2e(video_path, image_size, src_calib, dst_calib)
        save_frame_array(undistorted_frame_array, image_root, take_uid)

def process_video(take_uid, mp4_root, postfix_path, vrs_root, image_root):
    video_path = os.path.join(mp4_root, take_uid, postfix_path)
    vrs_path = os.path.join(vrs_root, take_uid, "aria01_noimagestreams.vrs")
    image_size, src_calib, dst_calib = parse_from_vrs(vrs_path, 448)
    _, undistorted_frame_array = undistort_video_e2e(video_path, image_size, src_calib, dst_calib)
    save_frame_array(undistorted_frame_array, image_root, take_uid)

def undistort_dataset_multi_process():
    take_uid_path = "data/metadata/take_uids.txt"
    with open(take_uid_path, "r") as f:
        take_uids = f.readlines()
    take_uids = [uid.strip() for uid in take_uids]
    print("Number of takes to process: ", len(take_uids))

    # video path
    mp4_root = "data/egoexo4d_v2/takes/"
    postfix_path = "frame_aligned_videos/downscaled/448/aria01_214-1.mp4"
    # VRS path
    vrs_root = "data/egoexo4d_v2/takes/"
    # save image path
    image_root = "data/egoexo4d_v2/processed_data/"

    with Pool() as p:
        p.starmap(process_video, [(take_uid, mp4_root, postfix_path, vrs_root, image_root) for take_uid in take_uids])

    return 0

def plot_image_compare(resized_frame, undistorted_frame, processed_image):
    plt.figure()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(resized_frame, cmap="gray", vmin=0, vmax=255)
    axes[0].title.set_text(f"sensor image")
    axes[0].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    axes[1].imshow(undistorted_frame, cmap="gray", vmin=0, vmax=255)
    axes[1].title.set_text(f"undistorted image")
    axes[1].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    axes[2].imshow(processed_image, cmap="gray", vmin=0, vmax=255)
    axes[2].title.set_text(f"processed image")
    axes[2].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    
    plt.savefig("undistorted_image.png")

def check_takeuid_align():
    # Get video path
    mp4_root = "data/egoexo4d_v2/takes/"
    mp4_take_uids = os.listdir(mp4_root)
    mp4_take_uids.sort()
    print("Number of takes from mp4: ", len(mp4_take_uids))
    # print("Video 0 take uid: ", mp4_take_uids[0])

    # Get VRS path
    vrs_root = "data/egoexo4d_v2/takes/"
    vrs_take_uids = os.listdir(vrs_root)
    vrs_take_uids.sort()
    print("Number of takes from vrs: ", len(vrs_take_uids))
    # print("VRS 0 take uid: ", vrs_take_uids[0])
    
    # Get processed path
    processed_v1_root = "data/egoexo4d_v1/processed_data/"
    processed_v1_uids = os.listdir(processed_v1_root)
    processed_v1_uids.sort()
    print("Number of takes from processed v1: ", len(processed_v1_uids))
    # print("Processed V1 take uid: ", processed_v1_uids[0])

    v2_mp4_take_uids = list(set(mp4_take_uids) - set(processed_v1_uids))
    v2_mp4_take_uids.sort()
    v2_vrs_take_uids = list(set(vrs_take_uids) - set(processed_v1_uids))
    v2_vrs_take_uids.sort()
    print("Number of takes from mp4 v2: ", len(v2_mp4_take_uids))
    print("Number of takes from vrs v2: ", len(v2_vrs_take_uids))

    # compute the intersection of take uids
    intersection = list(set(v2_mp4_take_uids) & set(v2_vrs_take_uids))
    intersection.sort()
    print("Number of intersection: ", len(intersection))

    # missing vrs
    missing_take_uids = list(set(v2_mp4_take_uids) - set(v2_vrs_take_uids))
    missing_take_uids.sort()

    return intersection, missing_take_uids

def get_total_take_uids():
    metadata_path = "data/metadata/takes.json"
    with open(metadata_path, "r") as f:
        takes = json.load(f)

    take_uids = []
    for take in tqdm(takes):
        take_uids.append(take["take_uid"])
    print("Number of take uids: ", len(take_uids))

    return take_uids

def undistort_dataset_args(start, end):
    take_uid_path = "data/metadata/take_uids.txt"
    with open(take_uid_path, "r") as f:
        take_uids = f.readlines()
    take_uids = [uid.strip() for uid in take_uids]
    take_uids.sort()
    take_uids = take_uids[start:end]
    print("Number of takes to process: ", len(take_uids))

    # video path
    mp4_root = "data/egoexo4d_v2/takes"
    postfix_path = "frame_aligned_videos/downscaled/448/" #aria01_214-1.mp4
    # VRS path
    vrs_root = "data/egoexo4d_v2/takes/"
    # save image path
    image_root = "data/egoexo4d_v2/processed_data/"

    for take_uid in tqdm(take_uids):
        video_dir = os.path.join(mp4_root, take_uid, postfix_path)
        # find file ends with "214-1.mp4"
        video_files = os.listdir(video_dir)
        for video_file in video_files:
            if video_file.endswith("214-1.mp4"):
                video_path = os.path.join(video_dir, video_file)
                break

        # find file ends with "noimagestreams.vrs"
        vrs_dir = os.path.join(vrs_root, take_uid)
        vrs_files = os.listdir(vrs_dir)
        for vrs_file in vrs_files:
            if vrs_file.endswith("noimagestreams.vrs"):
                vrs_path = os.path.join(vrs_dir, vrs_file)
                break

        if not video_path or not vrs_path:
            print("No valid files found for take_uid: ", take_uid)
            continue
        
        image_size, src_calib, dst_calib = parse_from_vrs(vrs_path, 448)
        _, undistorted_frame_array = undistort_video_e2e(video_path, image_size, src_calib, dst_calib)
        save_frame_array(undistorted_frame_array, image_root, take_uid)

def check_unprocessed():
    take_uid_path = "data/metadata/take_uids.txt"
    with open(take_uid_path, "r") as f:
        take_uids = f.readlines()
    take_uids = [uid.strip() for uid in take_uids]
    take_uids.sort()

    # save image path
    image_root = "data/egoexo4d_v2/processed_data/"
    processed_take_uids = os.listdir(image_root)
    processed_take_uids.sort()

    unprocessed_take_uids = list(set(take_uids) - set(processed_take_uids))
    unprocessed_take_uids.sort()
    print(unprocessed_take_uids)
    print(len(unprocessed_take_uids))

    return unprocessed_take_uids

def undistort_dataset_list(take_uids):
    take_uids.sort()
    print("Number of takes to process: ", len(take_uids))

    # video path
    mp4_root = "data/egoexo4d_v2/takes/"
    postfix_path = "frame_aligned_videos/downscaled/448/" #aria01_214-1.mp4
    # VRS path
    vrs_root = "data/egoexo4d_v2/takes/"
    # save image path
    image_root = "data/egoexo4d_v2/processed_data/"

    for take_uid in take_uids:
        video_dir = os.path.join(mp4_root, take_uid, postfix_path)
        # find file ends with "214-1.mp4"
        video_files = os.listdir(video_dir)
        for video_file in video_files:
            if video_file.endswith("214-1.mp4"):
                video_path = os.path.join(video_dir, video_file)
                break

        # find file ends with "noimagestreams.vrs"
        vrs_dir = os.path.join(vrs_root, take_uid)
        vrs_files = os.listdir(vrs_dir)
        for vrs_file in vrs_files:
            if vrs_file.endswith("noimagestreams.vrs"):
                vrs_path = os.path.join(vrs_dir, vrs_file)
                break

        if not video_path or not vrs_path:
            print("No valid files found for take_uid: ", take_uid)
            continue
        
        image_size, src_calib, dst_calib = parse_from_vrs(vrs_path, 448)
        _, undistorted_frame_array = undistort_video_e2e(video_path, image_size, src_calib, dst_calib)
        save_frame_array(undistorted_frame_array, image_root, take_uid)

def undistort_save_by_frame(aria_video_path, save_root, take_uid, src_calib, dst_calib):
    # read one frame
    # cap = cv2.VideoCapture(aria_video_path)
    # ret, frame = cap.read()
    # frame_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # cap.release()

    # read video
    cap = cv2.VideoCapture(aria_video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # make directory
    dir_path = os.path.join(save_root, take_uid, "aria01_214-1")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # count and save the first frame of every five
    pbar = tqdm(total=frame_count)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % 5 == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rectified_frame = calibration.distort_by_calibration(frame, dst_calib, src_calib, InterpolationMethod.BILINEAR)
            frame_path = os.path.join(dir_path, f"frame_{count}.jpg")
            frame = cv2.cvtColor(rectified_frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(frame_path, frame)
        count += 1
        pbar.update(1)
    cap.release()
    pbar.close()

    return 0

def undistort_dataset_list_by_frame(take_uids):
    take_uids.sort()
    print("Number of takes to process: ", len(take_uids))

    # video path
    mp4_root = "data/egoexo4d_v2/takes/"
    postfix_path = "frame_aligned_videos/downscaled/448/" #aria01_214-1.mp4
    # VRS path
    vrs_root = "data/egoexo4d_v2/takes/"
    # save image path
    image_root = "data/egoexo4d_v2/processed_data/"

    for take_uid in take_uids:
        video_dir = os.path.join(mp4_root, take_uid, postfix_path)
        # find file ends with "214-1.mp4"
        video_files = os.listdir(video_dir)
        for video_file in video_files:
            if video_file.endswith("214-1.mp4"):
                video_path = os.path.join(video_dir, video_file)
                break

        # find file ends with "noimagestreams.vrs"
        vrs_dir = os.path.join(vrs_root, take_uid)
        vrs_files = os.listdir(vrs_dir)
        for vrs_file in vrs_files:
            if vrs_file.endswith("noimagestreams.vrs"):
                vrs_path = os.path.join(vrs_dir, vrs_file)
                break

        if not video_path or not vrs_path:
            print("No valid files found for take_uid: ", take_uid)
            continue
        
        image_size, src_calib, dst_calib = parse_from_vrs(vrs_path, 448)
        undistort_save_by_frame(video_path, image_root, take_uid, src_calib, dst_calib)

def check_processed_length():
    take_uid_path = "intersection_take_uids.txt"
    with open(take_uid_path, "r") as f:
        take_uids = f.readlines()
    take_uids = [uid.strip() for uid in take_uids]
    take_uids.sort()

    # save image path
    image_root = "data/egoexo4d_v2/processed_data/"
    # video path
    mp4_root = "data/egoexo4d_v2/takes/"
    postfix_path = "frame_aligned_videos/downscaled/448/" #aria01_214-1.mp4

    broken_take_uids = []
    
    for take_uid in tqdm(take_uids):
        video_dir = os.path.join(mp4_root, take_uid, postfix_path)
        # find file ends with "214-1.mp4"
        video_files = os.listdir(video_dir)
        for video_file in video_files:
            if video_file.endswith("214-1.mp4"):
                video_path = os.path.join(video_dir, video_file)
                break
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved_frame_count = (frame_count-1) // 5 + 1
        cap.release()

        processed_dir = os.path.join(image_root, take_uid, "aria01_214-1")
        processed_files = os.listdir(processed_dir)
        processed_frame_count = len(processed_files)

        if saved_frame_count != processed_frame_count:
            broken_take_uids.append(take_uid)
            print(f"Take uid: {take_uid}, saved_frame_count: {saved_frame_count}, processed_frame_count: {processed_frame_count}")

    # save txt
    with open("broken_take_uids.txt", "w") as f:
        for uid in broken_take_uids:
            f.write(uid + "\n")

def find_similar_vrs(take_uid):
    # parse university and task
    university = take_uid.split("_")[0]
    # task should only be letters, exclude the numbers
    task = "".join([i for i in take_uid.split("_")[1] if not i.isdigit()])
    if university == "upenn":
        task = "".join([i for i in take_uid.split("_")[2] if not i.isdigit()])

    # VRS path
    vrs_root = "data/egoexo4d_v2/takes/"
    vrs_take_uids = os.listdir(vrs_root)
    similar_take_uid = vrs_take_uids[0]

    for vrs_take_uid in vrs_take_uids:
        if university in vrs_take_uid and task in vrs_take_uid:
            similar_take_uid = vrs_take_uid
            break
    
    similar_take_uid_university = similar_take_uid.split("_")[0]
    if similar_take_uid_university != university:
        for vrs_take_uid in vrs_take_uids:
            if university in vrs_take_uid:
                similar_take_uid = vrs_take_uid
                break        

    return similar_take_uid

def distort_using_similar_vrs():
    take_uid_path = "missing_take_uids.txt"
    with open(take_uid_path, "r") as f:
        take_uids = f.readlines()
    take_uids = [uid.strip() for uid in take_uids]
    take_uids.sort()
    print("Number of takes to process: ", len(take_uids))

    # video path
    mp4_root = "data/egoexo4d_v2/takes/"
    postfix_path = "frame_aligned_videos/downscaled/448/" #aria01_214-1.mp4
    # VRS path
    vrs_root = "data/egoexo4d_v2/takes/"
    # save image path
    image_root = "data/egoexo4d_v2/processed_data/"

    for take_uid in take_uids:
        video_dir = os.path.join(mp4_root, take_uid, postfix_path)
        # find file ends with "214-1.mp4"
        video_files = os.listdir(video_dir)
        for video_file in video_files:
            if video_file.endswith("214-1.mp4"):
                video_path = os.path.join(video_dir, video_file)
                break

        # find file ends with "noimagestreams.vrs"
        similar_take_uid = find_similar_vrs(take_uid)
        print("Replace the VRS file of "+str(take_uid)+" with "+str(similar_take_uid))
        vrs_dir = os.path.join(vrs_root, similar_take_uid)
        vrs_files = os.listdir(vrs_dir)
        for vrs_file in vrs_files:
            if vrs_file.endswith("noimagestreams.vrs"):
                vrs_path = os.path.join(vrs_dir, vrs_file)
                break

        if not video_path or not vrs_path:
            print("No valid files found for take_uid: ", take_uid)
            continue
        
        image_size, src_calib, dst_calib = parse_from_vrs(vrs_path, 448)
        undistort_save_by_frame(video_path, image_root, take_uid, src_calib, dst_calib)

def main(args):
    intersection_take_uids, missing_take_uids = check_takeuid_align()
    # save txt
    with open("intersection_take_uids.txt", "w") as f:
        for uid in intersection_take_uids:
            f.write(uid + "\n")
    with open("missing_take_uids.txt", "w") as f:
        for uid in missing_take_uids:
            f.write(uid + "\n")

    undistort_dataset_args(args.start, args.end)

    # undistort_dataset_multi_process()

    # unprocessed_take_uids = check_unprocessed()
    # unprocessed_take_uids = ["georgiatech_cooking_03_01_6", "georgiatech_cooking_04_03_2"]
    # undistort_dataset_list_by_frame(unprocessed_take_uids)

    # check_processed_length()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    args = parser.parse_args()

    main(args)
