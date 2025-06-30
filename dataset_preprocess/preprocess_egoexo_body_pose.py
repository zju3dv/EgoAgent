import os
import torch
import numpy as np

import json
from tqdm import tqdm
import random


root = "data/egoexo4d_v2"
save_dir = "data/egoexo4d_body_pose"
use_pseudo = True
split_all = ["train", "val"]
# Pose format
joint_idxs = [i for i in range(17)] # 17 keypoints in total
joint_names = ['nose','left-eye','right-eye','left-ear','right-ear','left-shoulder','right-shoulder','left-elbow','right-elbow','left-wrist','right-wrist','left-hip','right-hip','left-knee','right-knee','left-ankle','right-ankle']
slice_window =  5
coord = None

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def get_metadata_take(uid, metadata):
    for take in metadata:
        if take["take_uid"]==uid:
            return take


def translate_poses(anno, cams, coord):
        trajectory = {}
        to_remove = []
        for key in cams.keys():
            if "aria" in key:
                aria_key =  key
                break
        first = next(iter(anno))
        first_cam =  cams[aria_key]['camera_extrinsics'][first]
        T_first_camera = np.eye(4)
        T_first_camera[:3, :] = np.array(first_cam)
        for frame in anno:
            try:
                current_anno = anno[frame]
                current_cam =  cams[aria_key]['camera_extrinsics'][frame]
                T_world_camera_ = np.eye(4)
                T_world_camera_[:3, :] = np.array(current_cam)
                
                if coord == 'global':
                    T_world_camera = np.linalg.inv(T_world_camera_)
                elif coord == 'aria':
                    T_world_camera = np.dot(T_first_camera,np.linalg.inv(T_world_camera_))
                else:
                    T_world_camera = T_world_camera_
                assert len(current_anno) != 0 
                for idx in range(len(current_anno)):
                    joints = current_anno[idx]["annotation3D"]
                    for joint_name in joints:
                        joint4d = np.ones(4)
                        joint4d[:3] = np.array([joints[joint_name]["x"], joints[joint_name]["y"], joints[joint_name]["z"]])
                        if coord == 'global':
                            new_joint4d = joint4d
                        elif coord == 'aria':
                            new_joint4d = T_first_camera.dot(joint4d)
                        else:
                            new_joint4d = T_world_camera_.dot(joint4d) #The skels always stay in 0,0,0 wrt their camera frame
                        joints[joint_name]["x"] = new_joint4d[0]
                        joints[joint_name]["y"] = new_joint4d[1]
                        joints[joint_name]["z"] = new_joint4d[2]
                    current_anno[idx]["annotation3D"] = joints
                traj = T_world_camera[:3,3]
                trajectory[frame] = traj
            except:
                to_remove.append(frame)
            anno[frame] = current_anno
        keys_old = list(anno.keys())
        for frame in keys_old:
            if frame in to_remove:
                del anno[frame]
        return anno, trajectory


def process_body_pose(take_uid, split, root_poses, cameras, manually_annotated_takes, pseudo_annotated_takes):
    poses = []
    trajectories = []
    if take_uid+".json" in cameras:
        # read camera
        camera_json = json.load(open(os.path.join(root_poses.replace("body", "camera_pose"),take_uid+".json")))

        # read pose
        if use_pseudo and take_uid in pseudo_annotated_takes:
            pose_json = json.load(open(os.path.join(root_poses, "automatic", take_uid+".json")))
            if (len(pose_json) > (slice_window +2)) and split == "train":
                ann, traj = translate_poses(pose_json, camera_json, coord)
                if len(traj) > (slice_window +2):
                    poses = ann
                    trajectories = traj
            elif split != "train":
                ann, traj = translate_poses(pose_json, camera_json, coord)
                poses = ann
                trajectories = traj
        elif take_uid in manually_annotated_takes:
            pose_json = json.load(open(os.path.join(root_poses,"annotation",take_uid+".json")))
            if (len(pose_json) > (slice_window +2)) and split == "train":
                ann, traj = translate_poses(pose_json, camera_json, coord)
                if len(traj) > (slice_window +2):
                    poses = ann
                    trajectories = traj
            elif split != "train":
                ann, traj = translate_poses(pose_json, camera_json, coord)
                poses = ann
                trajectories = traj
        
        return poses, trajectories
    else:
        return None, None


def parse_skeleton(skeleton):
        poses = []
        flags = []
        keypoints = skeleton.keys()
        for keyp in joint_names:
            if keyp in keypoints:
                flags.append(1) #visible
                poses.append([skeleton[keyp]['x'], skeleton[keyp]['y'], skeleton[keyp]['z']]) #visible
            else:
                flags.append(0) #not visible
                poses.append([-1,-1,-1]) #not visible
        return poses, flags


# def save_json_in_fregments(take_motion, save_dir, take_uid):
#     # 100 frames per file, for the remaining frames, save in the last file
#     frames = list(take_motion.keys())
#     nframes = len(frames)
#     nfiles = nframes // 100
    
#     for i in range(nfiles):
#         start = i*100
#         end = (i+1)*100
#         file_name = f"{take_uid}_{frames[start]}-{frames[end]}.json"
#         with open(os.path.join(save_dir, file_name), "w") as f:
#             # Create a new dictionary with only the selected keys
#             selected_frames = {k: take_motion[k] for k in frames[start:end]}
#             json.dump(selected_frames, f)

#     if nframes % 100 != 0:
#         start = nfiles*100
#         file_name = f"{take_uid}_{frames[start]}-{frames[-1]}.json"
#         with open(os.path.join(save_dir, file_name), "w") as f:
#             # Create a new dictionary with only the selected keys
#             selected_frames = {k: take_motion[k] for k in frames[start:]}
#             json.dump(selected_frames, f)


def save_json_in_fregments(take_motion, save_dir, take_uid, window):
    # 100 frames per file, in which
    # the first 5 frames are the same as the last 5 frames of the previous file
    frames = list(take_motion.keys())
    nframes = len(frames)
    nfiles = nframes // 100
    
    for i in range(nfiles):
        start = i * 100 - window
        end = (i+1) * 100
        if i == 0:
            start = 0
        if i == nfiles - 1:
            end = nframes
        
        take_motion_frag = {}
        for j in range(start, end):
            take_motion_frag[frames[j]] = take_motion[frames[j]]
            
        file_name = f"{take_uid}_{frames[start]}-{frames[end-1]}.json"
        with open(os.path.join(save_dir, file_name), "w") as f:
            json.dump(take_motion_frag, f)


def main():
    for s in split_all:
        print(f"Processing {s} split")
        root_poses = os.path.join(root, "annotations", "ego_pose", s, "body")

        # Load manually annotated and pseudo annotated body pose data
        manually_annotated_takes = os.listdir(os.path.join(root_poses,"annotation"))
        manually_annotated_takes = [take.split(".")[0] for take in manually_annotated_takes]
        if use_pseudo:
            pseudo_annotated_takes = os.listdir(os.path.join(root_poses,"automatic"))
            pseudo_annotated_takes = [take.split(".")[0] for take in pseudo_annotated_takes]
        # Load camera pose
        cameras = os.listdir(root_poses.replace("body", "camera_pose"))
        
        # Load metadata
        metadata = json.load(open(os.path.join(root, "metadata", "takes.json")))
        takes_uids = pseudo_annotated_takes if use_pseudo else manually_annotated_takes
        takes_metadata = {}

        for take_uid in takes_uids:
            take_temp = get_metadata_take(take_uid, metadata)
            if take_temp and 'bouldering' not in take_temp['take_name']:
                takes_metadata[take_uid] = take_temp

        poses = {}
        trajectories = {}

        # Process body pose and trajectory
        for take_uid in tqdm(takes_metadata):
            take_motion = {}
            poses, trajectories = process_body_pose(take_uid, s, root_poses, cameras, manually_annotated_takes, pseudo_annotated_takes)
            if poses is None:
                continue
            capture_frames =  list(poses.keys())

            for frame in capture_frames:
                pose, flags = parse_skeleton(poses[frame][0]["annotation3D"]) # list
                trajectory = trajectories[frame].tolist() # list

                take_motion[frame] = {"pose": pose, "confidence": flags, "trajectory": trajectory}
                # take_motion.append({frame: {"pose": pose, "confidence": flags, "trajectory": trajectory}})

            save_dir_split = os.path.join(save_dir, s)
            if not os.path.exists(save_dir_split):
                os.makedirs(save_dir_split)
            # np.save(os.path.join(save_dir_split, f"{take_uid}.npy"), take_motion)
            # with open(os.path.join(save_dir_split, f"{take_uid}.json"), "w") as f:
            #     json.dump(take_motion, f)
            save_json_in_fregments(take_motion, save_dir_split, take_uid, 5)
    

if __name__ == "__main__":
    main()