import cv2
#vidcap = cv2.VideoCapture('G:\\[4K] [미스터 로드뷰 Mr. Road View] 47화.mp4')
#vidcap = cv2.VideoCapture('\\\\172.26.19.199\\dataset\\Younkwan\\CVPR20_AICITY\\AIC20_track1_vehicle_counting\\AIC20_track1\\counting_gt_sample\\counting_example_cam_5_1min.mp4')
#vidcap = cv2.VideoCapture('C:\\Users\\MLV\\Documents\\oCam\\nh_001.mp4')
vidcap = cv2.VideoCapture('C:\\Users\\MLV\\Desktop\\10lx_clean.mp4')
vidcap = cv2.VideoCapture('C:\\Users\\MLV\\Desktop\\10lx_68dB_76dB.mp4')
vidcap = cv2.VideoCapture('I:\\(15min)100-120dB.mp4')
vidcap = cv2.VideoCapture('I:\\(18min)Original.mp4')
#vidcap = cv2.VideoCapture('I:\\서면교\\ground_truth\\서면교_주간.mp4')
vidcap = cv2.VideoCapture('I:\\서면교\\night_truth\\서면교_야간.avi')

success,image = vidcap.read()
count = 0
success = True

while success:
    if count % 1000 ==0:
        #cv2.imwrite("./Dataset/DW_004_case3/frame%d.jpg" % (count), image)  # save frame as JPEG file
        #cv2.imwrite("./Dataset/200717/ground_truth/frame%d.jpg" % (count), image)  # save frame as JPEG file
        cv2.imwrite("./Dataset/200727/rainy_image/frame%d.jpg" % (count), image)  # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success, count)
    count += 1
    #if count / 20 == 0:
