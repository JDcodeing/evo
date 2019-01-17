import os
import re
import numpy as np
import csv
import numpy.matlib as ml
from transform import build_se3_transform
import copy

from interpolate_poses import interpolate_vo_poses, interpolate_ins_poses, interpolate_poses



def compute_ins_poses(ins_path, pose_timestamps, origin_timestamp):
    """Interpolate poses from visual odometry.

    ins_path (str): path to file containing poses from INS.
        pose_timestamps (list[int]): UNIX timestamps at which interpolated poses are required.
        origin_timestamp (int): UNIX timestamp of origin frame. Poses will be reported relative to this frame.

    Returns:
        list[numpy.matrixlib.defmatrix.matrix]: SE3 matrix representing interpolated pose for each requested timestamp.

    """
    with open(ins_path) as ins_file:
        ins_reader = csv.reader(ins_file)
        headers = next(ins_file)

        ins_timestamps = [0]
        abs_poses = [ml.identity(4)]
        #abs_poses[0] = np.dot(changetoDSOcoord, abs_poses[0])

        lower_timestamp = min(min(pose_timestamps), origin_timestamp)
        upper_timestamp = max(max(pose_timestamps), origin_timestamp)

        for row in ins_reader:
            timestamp = int(row[0])
            if timestamp < lower_timestamp:
                ins_timestamps[0] = timestamp
                continue
            ins_timestamps.append(timestamp)

            xyzrpy = [float(v) for v in row[5:8]] + [float(v) for v in row[-3:]]
            abs_pose = build_se3_transform(xyzrpy)
            abs_poses.append(abs_pose)

            if timestamp >= upper_timestamp:
                break

    ins_timestamps = ins_timestamps[1:]
    abs_poses = abs_poses[1:]
    print(len(pose_timestamps))
    print("ins timestamps",len(ins_timestamps))
    #poses = interpolate_poses(ins_timestamps, abs_poses, pose_timestamps, origin_timestamp)
    #print(len(poses))
    changetoDSOcoord = np.matrix('1 0 0 0; 0 0 -1 0; 0 1 0 0; 0 0 0 1')
    poses_trans = [ changetoDSOcoord*i for i in abs_poses]

    return abs_poses,ins_timestamps
    

def GetGTPoses_Ins(ins_file, timestamps_path, start_time, end_time, origin_time=-1):
    """Builds a pointcloud by combining multiple LIDAR scans with odometry information.

    Args:
        ins_file (str): Directory containing ins scans.
        timestamps_path (str): Path to a file containing timestamps information.
        start_time (int): UNIX timestamp of the start of the window over which to build the pointcloud.
        end_time (int): UNIX timestamp of the end of the window over which to build the pointcloud.
        origin_time (int): UNIX timestamp of origin frame. Pointcloud coordinates are relative to this frame.

    Returns:
        

    Raises:
        ValueError: if specified window doesn't contain any laser scans.
        IOError: if scan files are not found.

    """
    if origin_time < 0:
        origin_time = start_time

    #lidar = re.search('(lms_front|lms_rear|ldmrs)', lidar_dir).group(0)
    #timestamps_path = os.path.join(lidar_dir, os.pardir, lidar + '.timestamps')

    timestampsreq = []
    with open(timestamps_path) as timestamps_file:
        for line in timestamps_file:
            timestamp = int(line.split(' ')[0])
            if start_time <= timestamp <= end_time:
                timestampsreq.append(timestamp)

    if len(timestampsreq) == 0:
        raise ValueError("No data in the given time bracket.")

    poses, instimes = compute_ins_poses(ins_file, timestampsreq, origin_time)

    return poses,timestampsreq,instimes

if __name__ == "__main__":
    import argparse


    datasetfile = "2014-05-19-13-05-38"
    #lidar = re.search('(lms_front|lms_rear|ldmrs)', args.laser_dir).group(0)
    #vo_file = "/usr/data/cvpr_shared/RobotCar/"+datasetfile+"/vo/vo.csv"
    ins_file = "/Users/Azalea/CLionProjects/data/robotcar/"+datasetfile+"/gps/ins.csv"
    timestamps_path = "/Users/Azalea/CLionProjects/data/robotcar/"+datasetfile+"/stereo.timestamps" #os.path.join(args.laser_dir, os.pardir, lidar + '.timestamps')
    savepath = "/Users/Azalea/CLionProjects/data/robotcar/"+datasetfile+"/gps/gt_ins.txt"
    with open(timestamps_path) as timestamps_file:
        for i, line in enumerate(timestamps_file):
            if i == 0:
                start_time = int(line.split(' ')[0])
    end_time = start_time + 2e12 
    poses, stamps, instimes = GetGTPoses_Ins(ins_file, timestamps_path, start_time, end_time)

    print("the size of poses :", len(poses))
    flattened = np.squeeze([x.flatten() for x in poses])
    #arr = np.ndarray(flattened)
    print(flattened.shape)
    print(len(stamps))
    print(len(instimes))
    stamps = np.array(stamps)
    gt = np.c_[instimes, flattened]
    print(gt.shape)
    np.savetxt(savepath, gt[:,0:13], delimiter=' ',fmt='%1.6f')
