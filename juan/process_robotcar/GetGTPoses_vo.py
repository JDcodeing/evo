import os
import re
import numpy as np
import csv
import numpy.matlib as ml
from transform import build_se3_transform
import copy

from interpolate_poses import interpolate_vo_poses, interpolate_ins_poses


def compute_vo_poses(vo_path, pose_timestamps, origin_timestamp):
    """Interpolate poses from visual odometry.

    Args:
        vo_path (str): path to file containing relative poses from visual odometry.
        pose_timestamps (list[int]): UNIX timestamps at which interpolated poses are required.
        origin_timestamp (int): UNIX timestamp of origin frame. Poses will be reported relative to this frame.

    Returns:
        list[numpy.matrixlib.defmatrix.matrix]: SE3 matrix representing interpolated pose for each requested timestamp.

    """
    with open(vo_path) as vo_file:
        vo_reader = csv.reader(vo_file)
        headers = next(vo_file)

        vo_timestamps = [0]

        changetoDSOcoord = np.matrix('0 1 0 0; 0 0 -1 0; 1 0 0 0; 0 0 0 1')
        #changetoDSOcoord = changetoDSOcoord.transpose()
        #changetoDSOcoord = np.matrix('0 0 1 0; 1 0 0 0; 0 1 0 0; 0 0 0 1')
        #changetoDSOcoord = np.matrix('0 0 1 0; 0 1 0 0; 1 0 0 0; 0 0 0 1')  false
        #changetoDSOcoord = np.matrix('0 1 0 0; 1 0 0 0; 0 0 1 0; 0 0 0 1') false
        #changetoDSOcoord = np.matrix('0 0 1 0; -1 0 0 0; 0 1 0 0; 0 0 0 1')
        #R2 = np.matrix('0 1 0; 0 0 1; 1 0 0')
        abs_poses = [ml.identity(4)]
        abs_poses[0] = np.dot(changetoDSOcoord, abs_poses[0])

        lower_timestamp = min(min(pose_timestamps), origin_timestamp)
        upper_timestamp = max(max(pose_timestamps), origin_timestamp)
        print("lower timestamp, ", lower_timestamp)
        print("upper timestamp, ", upper_timestamp)
        for row in vo_reader:
            timestamp = int(row[0])
            if timestamp < lower_timestamp:
                vo_timestamps[0] = timestamp
                continue

            vo_timestamps.append(timestamp)

            xyzrpy = [float(v) for v in row[2:8]]
            rel_pose = build_se3_transform(xyzrpy)
            abs_pose = abs_poses[-1] * rel_pose
            #changetoDSOcoord = np.matrix('0 1 0 0; 0 0 -1 0; 1 0 0 0; 0 0 0 1')
            #abs_pose = np.dot(changetoDSOcoord,abs_pose)
            #abs_posecopy = copy.deepcopy(abs_pose)
            #abs_posecopy[0,3],abs_posecopy[1,3],abs_posecopy[2,3]=abs_pose[2,3],abs_pose[0,3],abs_pose[1,3]
            abs_poses.append(abs_pose)


            if timestamp >= upper_timestamp:
                break
        abs_poses[0] = np.dot(changetoDSOcoord,abs_poses[0])


    return abs_poses

def GetGTPoses_vo(vo_file, timestamps_path, start_time, end_time, origin_time=-1):
    """Builds a pointcloud by combining multiple LIDAR scans with odometry information.

    Args:
        lidar_dir (str): Directory containing LIDAR scans.
        poses_file (str): Path to a file containing pose information. Can be VO or INS data.
        extrinsics_dir (str): Directory containing extrinsic calibrations.
        start_time (int): UNIX timestamp of the start of the window over which to build the pointcloud.
        end_time (int): UNIX timestamp of the end of the window over which to build the pointcloud.
        origin_time (int): UNIX timestamp of origin frame. Pointcloud coordinates are relative to this frame.

    Returns:
        numpy.ndarray: 3xn array of (x, y, z) coordinates of pointcloud
        numpy.array: array of n reflectance values or None if no reflectance values are recorded (LDMRS)

    Raises:
        ValueError: if specified window doesn't contain any laser scans.
        IOError: if scan files are not found.

    """
    if origin_time < 0:
        origin_time = start_time

    #lidar = re.search('(lms_front|lms_rear|ldmrs)', lidar_dir).group(0)
    #timestamps_path = os.path.join(lidar_dir, os.pardir, lidar + '.timestamps')

    timestamps = []
    with open(timestamps_path) as timestamps_file:
        for line in timestamps_file:
            timestamp = int(line.split(' ')[0])
            if start_time <= timestamp <= end_time:
                timestamps.append(timestamp)

    if len(timestamps) == 0:
        raise ValueError("No LIDAR data in the given time bracket.")


    poses = compute_vo_poses(vo_file, timestamps, origin_time)

    return poses

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    datasetfile = "2014-05-06-12-54-54"
    #lidar = re.search('(lms_front|lms_rear|ldmrs)', args.laser_dir).group(0)
    #vo_file = "/usr/data/cvpr_shared/RobotCar/"+datasetfile+"/vo/vo.csv"
    vo_file = "/Users/Azalea/CLionProjects/data/robotcar/"+datasetfile+"/vo/vo.csv"
    timestamps_path = "/Users/Azalea/CLionProjects/data/robotcar/"+datasetfile+"/stereo.timestamps" #os.path.join(args.laser_dir, os.pardir, lidar + '.timestamps')
    savepath = "/Users/Azalea/CLionProjects/data/robotcar/"+datasetfile+"/vo/gt_trans.txt"
    with open(timestamps_path) as timestamps_file:
        for i, line in enumerate(timestamps_file):
            if i == 0:
                start_time = int(line.split(' ')[0])
    #start_time = 1399382134674366 # best 1417794411042204 #1417794411167202 #2400 1417794410979712#1800 1417794369110314# 600 1417794220193019# 01417794166325288

    end_time = start_time + 2e12
    #end_time = 1417794607640875
    poses = GetGTPoses_vo(vo_file, timestamps_path, start_time, end_time)

    print("the size of poses :", len(poses))
    flattened = np.squeeze([x.flatten() for x in poses])
    #arr = np.ndarray(flattened)
    print(flattened.shape)
    #print(flattened[0:3])
    np.savetxt(savepath, flattened[:,0:12], delimiter=' ',fmt='%1.6e')
