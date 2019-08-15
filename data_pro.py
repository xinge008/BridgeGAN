import os
import shutil

des_dir_homo = "/media/sensetime/data/unit/day_homo/"
des_dir_gt = "/media/sensetime/data/unit/day_gt/"

des_dir_homo_file = "/media/sensetime/data/unit/day_homo_list.txt"
des_dir_gt_file = "/media/sensetime/data/unit/day_gt_list.txt"


old_dir = "/media/sensetime/data/GTAV_day_nocar/"

path_dict = {}

def get_path_dict():
    for subdir1 in os.listdir(old_dir):
        subdir2 = os.path.join(old_dir, subdir1)
        for subfile in os.listdir(subdir2):
            key1 = subdir1+"_"+subfile[0:6]
            path_dict[key1] = 0

    print len(path_dict)
    # print path_dict.popitem()

def copy_to_homo_and_gt():
    w1 = open(des_dir_gt_file,'wb')
    w2 = open(des_dir_homo_file, 'wb')
    for key in path_dict.keys():
        sub = key.split("_")
        homo_path = os.path.join(old_dir, sub[0], sub[1]+"_homo.jpg")
        gt_path = os.path.join(old_dir, sub[0],sub[1]+"_gt.jpg")
        tar_homo_path = os.path.join(des_dir_homo, key+".jpg")
        tar_gt_path = os.path.join(des_dir_gt, key+"_b.jpg")
        w1.write(key+".jpg\n")
        w2.write(key+"_b.jpg\n")
        shutil.copyfile(homo_path, tar_homo_path)
        shutil.copyfile(gt_path, tar_gt_path)
        print key


if __name__ == '__main__':
    get_path_dict()
    copy_to_homo_and_gt()