import os
from tqdm import tqdm

path = "missing_take_uids.txt"
with open(path, "r") as f:
    missing_take_uids = f.readlines()
missing_take_uids = [uid.strip() for uid in missing_take_uids]
print("Number of missing take uids: ", len(missing_take_uids))

# download missing vrs
for take_uid in tqdm(missing_take_uids):
    cmd = f"egoexo -o data/egoexo4d_v2/missing_vrs/ --uids {take_uid} --release v2 --s3_profile egoexo4d"
    os.system(cmd)
    print("Downloaded missing vrs: ", take_uid)