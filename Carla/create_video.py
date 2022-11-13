# from asyncio.windows_events import NULL
import cv2
import numpy as np
import glob
import argparse
import tqdm
import os

TOTAL_RUNS = 11
argparser = argparse.ArgumentParser()
argparser.add_argument(
        '-n', '--run_no',
        metavar='P',
        default=-1,
        type=int,
        help='Run no')
args = argparser.parse_args()

if args.run_no != -1 :
    TOTAL_RUNS = args.run_no

if not os.path.exists('Videos/'):
    os.makedirs('Videos/')

for i in tqdm.tqdm(range(1,TOTAL_RUNS+1)) : 
    RUN_NO = i
    INPUT_FOLDER = 'with_cbf_dynamic_updated/run'+str(RUN_NO)+'_video'
    OUTPUT_FILE = 'Videos/run'+str(RUN_NO)+'_video.mp4'
    img_array = []
    # print(INPUT_FOLDER+'/*')
    file_list = glob.glob(INPUT_FOLDER+'/*')
    n_files = len(file_list)
    file_list.sort()
    file_list = []
    for i in range(16,n_files+16) :
        file_list.append(INPUT_FOLDER+'/frame_'+str(i)+'.png')
    # print(file_list)
    for filename in file_list:
        img = cv2.imread(filename)
        # if img != None:
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)


    out = cv2.VideoWriter(OUTPUT_FILE,cv2.VideoWriter_fourcc(*'mp4v'), 25, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


