import sys
import subprocess
import cv2
import time
import os
import argparse
import tensorflow as tf
import ctc_utils
import cv2
import shutil
import zipfile
import numpy as np
from best_fit import fit
from rectangle import Rectangle
from note import Note
from random import randint
from os.path import basename

staff_files = [
    "resources/template/staff2.png", 
    "resources/template/staff.png"]

staff_imgs = [cv2.imread(staff_file, 0) for staff_file in staff_files]

staff_lower, staff_upper, staff_thresh = 50, 150, 0.5

def locate_images(img, templates, start, stop, threshold):
    locations, scale = fit(img, templates, start, stop, threshold)
    img_locations = []
    for i in range(len(templates)):
        w, h = templates[i].shape[::-1]
        w *= scale
        h *= scale
        img_locations.append([Rectangle(pt[0], pt[1], w, h) for pt in zip(*locations[i][::-1])])
    return img_locations

def merge_recs(recs, threshold):
    filtered_recs = []
    while len(recs) > 0:
        r = recs.pop(0)
        recs.sort(key=lambda rec: rec.distance(r))
        merged = True
        while(merged):
            merged = False
            i = 0
            for _ in range(len(recs)):
                if r.overlap(recs[i]) > threshold or recs[i].overlap(r) > threshold:
                    r = r.merge(recs.pop(i))
                    merged = True
                elif recs[i].distance(r) > r.w/2 + recs[i].w/2:
                    break
                else:
                    i += 1
        filtered_recs.append(r)
    return filtered_recs

def open_file(path):
    cmd = {'linux':'eog', 'win32':'explorer', 'darwin':'open'}[sys.platform]
    subprocess.run([cmd, path])

def zipfolder(path, zip_file):            
    zipobj = zipfile.ZipFile(foldername + '.zip', 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            fn = os.path.join(base, file)
            zipobj.write(fn, fn[rootlen:])

if __name__ == "__main__":
    img_file = sys.argv[1:][0]
    in_path = 'in/' + img_file
    img = cv2.imread(in_path, 0)
    img_gray = img#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
    ret,img_gray = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
    img_width, img_height = img_gray.shape[::-1]

    print("Matching staff image...")
    staff_recs = locate_images(img_gray, staff_imgs, staff_lower, staff_upper, staff_thresh)

    print("Filtering weak staff matches...")
    staff_recs = [j for i in staff_recs for j in i]
    heights = [r.y for r in staff_recs] + [0]
    histo = [heights.count(i) for i in range(0, max(heights) + 1)]
    avg = np.mean(list(set(histo)))
    staff_recs = [r for r in staff_recs if histo[r.y] > avg]

    print("Merging staff image results...")
    staff_recs = merge_recs(staff_recs, 0.01)
    staff_recs_img = img.copy()
    for r in staff_recs:
        r.draw(staff_recs_img, (0, 0, 255), 2)
#    cv2.imwrite('staff_recs_img2.png', staff_recs_img)
#    open_file('staff_recs_img2.png')

    print("Discovering staff locations...")
    recs = [Rectangle(r.x, r.y, r.w, r.h)for r in staff_recs]
    theWidth = 0
    for i in range(len(recs)):
        theWidth=max(theWidth,recs[i].x+recs[i].w)

    staff_boxes = merge_recs([Rectangle(r.x, r.y, img_width, r.h) for r in staff_recs], 0.01)
    staff_boxes_img = img.copy()
    i = 0
    # define the name of the directory to be created
    out_path = 'out/' + img_file[:20]

    try:  
        os.mkdir(out_path)
    except OSError:  
        print ("Creation of the directory %s failed" % out_path)
    else:  
        print ("Successfully created the directory %s " % out_path)

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    # Read the dictionary
    dict_file = open('Data/vocabulary_semantic.txt','r')
    dict_list = dict_file.read().splitlines()
    int2word = dict()
    for word in dict_list:
        word_idx = len(int2word)
        int2word[word_idx] = word
    dict_file.close()

    # model_name = './Models/semantic_model.meta'
    # Restore weights
    saver = tf.train.import_meta_graph('./Models/semantic_model.meta')
    saver.restore(sess,'./Models/semantic_model.meta'[:-5])

    graph = tf.get_default_graph()

    input = graph.get_tensor_by_name("model_input:0")
    seq_len = graph.get_tensor_by_name("seq_lengths:0")
    rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
    height_tensor = graph.get_tensor_by_name("input_height:0")
    width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
    logits = tf.get_collection("logits")[0]

    # Constants that are saved inside the model itself
    WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

    decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)

    for r in staff_boxes:
        r.w=theWidth-r.x
        crop_img = staff_boxes_img[r.y:r.y+int(r.h), r.x:r.x+int(r.w)]
        cv2.imwrite(out_path + '/staff_boxes_img' + str(i) + '.png', crop_img)
        image = cv2.imread(out_path + '/staff_boxes_img' + str(i) + '.png',False)
        image = ctc_utils.resize(image, HEIGHT)
        image = ctc_utils.normalize(image)
        image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

        seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]

        prediction = sess.run(decoded,
                              feed_dict={
                                  input: image,
                                  seq_len: seq_lengths,
                                  rnn_keep_prob: 1.0,
                              })

        str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)

        f = open(out_path + '/' + img_file[:20] + '_' + str(i) + '.txt', 'w')
        for w in str_predictions[0]:
            f.write(int2word[w] + '\t')

        f.close()
        i += 1
#        open_file('staff_boxes_img2.png')
    zipf = zipfile.ZipFile('out/' + img_file[:20] + '.zip', 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(out_path):  
        for filename in files:
            zipf.write(os.path.join(root, filename), basename(os.path.join(root, filename)))
    zipf.close()

