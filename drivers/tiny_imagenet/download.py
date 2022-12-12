# This code based on https://github.com/lokeshpara/Freecodecamp/tree/master/course_project


import os
import urllib.request
import zipfile
from random import shuffle
from math import floor

def download_dataset():
    print('Beginning dataset download with urllib2')
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    path = "%s/tiny-imagenet-200.zip" % os.getcwd()
    urllib.request.urlretrieve(url, path)
    print("Dataset downloaded")

def unzip_data():
    path_to_zip_file = "%s/tiny-imagenet-200.zip" % os.getcwd()
    directory_to_extract_to = os.getcwd()
    print("Extracting zip file: %s" % path_to_zip_file)
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    print("Extracted at: %s" % directory_to_extract_to)

def format_val():
    val_dir = "%s/tiny-imagenet-200/val" % os.getcwd()
    print("Formatting: %s" % val_dir)
    val_annotations = "%s/val_annotations.txt" % val_dir
    val_dict = {}
    with open(val_annotations, 'r') as f:
        for line in f:
            line = line.strip().split()
            assert(len(line) == 6)
            wnind = line[1]
            img_name = line[0]
            boxes = '\t'.join(line[2:])
            if wnind not in val_dict:
                val_dict[wnind] = []
            entries = val_dict[wnind]
            entries.append((img_name, boxes))
    assert(len(val_dict) == 200)
    for wnind, entries in val_dict.items():
        val_wnind_dir = "%s/%s" % (val_dir, wnind)
        val_images_dir = "%s/images" % val_dir
        val_wnind_images_dir = "%s/images" % val_wnind_dir
        os.mkdir(val_wnind_dir)
        os.mkdir(val_wnind_images_dir)
        wnind_boxes = "%s/%s_boxes.txt" % (val_wnind_dir, wnind)
        f = open(wnind_boxes, "w")
        for img_name, box in entries:
            source = "%s/%s" % (val_images_dir, img_name)
            dst = "%s/%s" % (val_wnind_images_dir, img_name)
            os.system("cp %s %s" % (source, dst))
            f.write("%s\t%s\n" % (img_name, box))
        f.close()
    os.system("rm -rf %s" % val_images_dir)
    print("Cleaning up: %s" % val_images_dir)
    print("Formatting val done")

def split_train_test():
    split_quota = 0.8
    print("Splitting Train+Val into %s-%s" % (split_quota*100, (1 - split_quota)*100))
    base_dir = "%s/tiny-imagenet-200" % os.getcwd()
    train_dir = "%s/train" % base_dir
    val_dir = "%s/val" % base_dir
    fwnind = "%s/wnids.txt" % base_dir
    wninds = set()
    with open(fwnind, "r") as f:
        for wnind in f:
            wninds.add(wnind.strip())
    assert(len(wninds) == 200)
    new_train_dir = "%s/new_train" % base_dir
    new_test_dir = "%s/new_test" % base_dir
    os.mkdir(new_train_dir)
    os.mkdir(new_test_dir)
    total_ntrain = 0
    total_ntest = 0
    for wnind in wninds:
        wnind_ntrain = 0
        wnind_ntest = 0
        new_train_wnind_dir = "%s/%s" % (new_train_dir, wnind)
        new_test_wnind_dir = "%s/%s" % (new_test_dir, wnind)
        os.mkdir(new_train_wnind_dir)
        os.mkdir(new_test_wnind_dir)
        os.mkdir(new_train_wnind_dir+"/images")
        os.mkdir(new_test_wnind_dir+"/images")
        new_train_wnind_boxes = "%s/%s_boxes.txt" % (new_train_wnind_dir, wnind)
        f_ntrain = open(new_train_wnind_boxes, "w")
        new_test_wnind_boxes = "%s/%s_boxes.txt" % (new_test_wnind_dir, wnind)
        f_ntest = open(new_test_wnind_boxes, "w")
        dirs = [train_dir, val_dir]
        for wdir in dirs:
            wnind_dir = "%s/%s" % (wdir, wnind)
            wnind_boxes = "%s/%s_boxes.txt" % (wnind_dir, wnind)
            imgs = []
            with open(wnind_boxes, "r") as f:
                for line in f:
                    line = line.strip().split()
                    img_name = line[0]
                    boxes = '\t'.join(line[1:])
                    imgs.append((img_name, boxes))
            print("[Old] wind: %s - #: %s" % (wnind, len(imgs)))
            shuffle(imgs)
            split_n = floor(len(imgs)*0.8)
            train_imgs = imgs[:split_n]
            test_imgs = imgs[split_n:]
            for img_name, box in train_imgs:
                source = "%s/images/%s" % (wnind_dir, img_name)
                dst = "%s/images/%s" % (new_train_wnind_dir, img_name)
                os.system("cp %s %s" % (source, dst))
                f_ntrain.write("%s\t%s\n" % (img_name, box))
                wnind_ntrain += 1
            for img_name, box in test_imgs:
                source = "%s/images/%s" % (wnind_dir, img_name)
                dst = "%s/images/%s" % (new_test_wnind_dir, img_name)
                os.system("cp %s %s" % (source, dst))
                f_ntest.write("%s\t%s\n" % (img_name, box))
                wnind_ntest += 1
        f_ntrain.close()
        f_ntest.close()
        print("[New] wnind: %s - #train: %s - #test: %s" % (wnind, wnind_ntrain,
                                                            wnind_ntest))
        total_ntrain += wnind_ntrain
        total_ntest += wnind_ntest
    print("[New] #train: %s - #test: %s" % (total_ntrain, total_ntest))
    os.system("rm -rf %s" % train_dir)
    os.system("rm -rf %s" % val_dir)
    print("Cleaning up: %s" % train_dir)
    print("Cleaning up: %s" % val_dir)
    print("Created new train data at: %s" % new_train_dir)
    print("Cleaning new test data at: %s" % new_test_dir)
    print("Splitting dataset done")


##
# Begin script
download_dataset()
unzip_data()
format_val()
split_train_test()

##
# Clean up Zip file
os.system("mv " + "%s/tiny-imagenet-200.zip ./tiny-imagenet-200" % os.getcwd() )

