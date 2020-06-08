#!/usr/bin/python3

import time
import cv2
import math, numpy as np
import socket
import sys, os
import zlib
import threading

PRESET_IMG_SIZE = (int(2560 * 0.5*0.25), int(960 * 1*0.25)) #set output resolution must same as reciever
PRESET_IMG_SECTION_NUM  = 16 # must same as reciever
PRESET_IMG_QUALITY  = 80    # 0 - 100, default = 95

SOCK_CONN   = None
SOCKET_HOST = "localhost"
SOCKET_PORT = 1112
PROTOCOL_DATA_DELIMITER = b"[HEADER]"
SOCKET_BUFF_SIZE = 20480

_FONT_HUD_  = cv2.FONT_HERSHEY_SIMPLEX

print(sys.argv)
if len(sys.argv) >= 3:
    SOCKET_HOST = str(sys.argv[2])

print("[START] in {}".format(os.getcwd()))

print("Initialize socket {}:{} ...".format(SOCKET_HOST, SOCKET_PORT))
SOCK_CONN   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
SOCK_CONN.settimeout(10)

# set socket option
bufsize = SOCK_CONN.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
print("Socket sending buffer size = ", bufsize)
bufsize = SOCK_CONN.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
print("Socket receiving buffer size = ", bufsize)

print("My sending buffer size = ", SOCKET_BUFF_SIZE)

# START

#region Stream Data

if len(sys.argv) >= 3:

    # capture webcam
    # cap = cv2.VideoCapture(int(sys.argv[1]))
    cap = cv2.imread('./dataset/img_test.png')
# else:

    # capture video file
    # cap = cv2.VideoCapture("../src/video.mp4")
    
try:

    t_begin = 0
    t_end   = 0
    t_fps   = 0

    t_display_data  = 0

    counter_error   = 0

    # while(cap.isOpened()):
    while(1):##############
        t_begin = time.perf_counter()

        frame = cap.copy()#################
        
        # ret, frame  = cap.read()
        # if ret is False or ret is None:
        #     counter_error   += 1
        #     # exit
        #     if counter_error > 30:
        #         print("Error limited : Read data fail !")
        #         break
        #     else:
        #         print("Restart to frame 0")
        #         cap = cv2.VideoCapture("../src/video.mp4")
        #         continue

        counter_error   = 0

        _IMG_HEIGHT_, _IMG_WIDTH_ = frame.shape[:2]

        # Convert
        # img_full    = frame # bypass conversion
        img_full        = cv2.resize(frame, PRESET_IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        # img_full_cvt    = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
        img_full_cvt    = img_full  # bypass conversion
        _IMG_HEIGHT_, _IMG_WIDTH_ = img_full_cvt.shape[:2]

        # t_debug_1   = time.perf_counter()

        # slice/cut image into package
        for package in range(PRESET_IMG_SECTION_NUM):
            img_height_per_pkg  = int(_IMG_HEIGHT_ / PRESET_IMG_SECTION_NUM)
            img_py_start        = int(package * img_height_per_pkg)
            img_slice           = img_full_cvt[img_py_start:img_py_start + img_height_per_pkg, 0:_IMG_WIDTH_]

            # Encode
            # img_send_raw    = img_slice.ravel() # convert to raw (1d)
            img_send_jpeg = cv2.imencode(".jpg", img_slice, [cv2.IMWRITE_JPEG_QUALITY, PRESET_IMG_QUALITY])[1] # convert to jpg
            # img_send_gzip   = zlib.compress(img_slice) # convert to gzip

            # img_len = len(img_send_raw)   # data size : raw
            img_len = len(img_send_jpeg)   # data size : jpeg
            # img_len = len(img_send_gzip)   # data size : gzip
            # print("section({}) to {}:{} = total {} byte".format(package, SOCKET_HOST, SOCKET_PORT, img_len))

            # Stream
            HEADER_PKG_NUM = PROTOCOL_DATA_DELIMITER + bytes([package])
            SOCK_CONN.sendto(HEADER_PKG_NUM, (SOCKET_HOST, SOCKET_PORT))   # send HEADER
            package_num = math.ceil(img_len / SOCKET_BUFF_SIZE)

            for x in range(package_num):
                i_start = x * SOCKET_BUFF_SIZE
                i_end   = i_start + SOCKET_BUFF_SIZE
                # data_to_send    = img_send_raw[i_start:i_end]   # send raw
                data_to_send    = img_send_jpeg[i_start:i_end]   # send jpeg
                # data_to_send    = img_send_gzip[i_start:i_end]   # send gzip
                SOCK_CONN.sendto(data_to_send, (SOCKET_HOST, SOCKET_PORT))
                #cv2.waitKey(1)  # delay for debug

                print("package({}) = {} byte".format(x, len(data_to_send)))


        # t_debug_2   = time.perf_counter()
        # print("t_debug_1={}".format(t_debug_2 - t_debug_1))

        # Draw
        tmp_text    = str(round(t_fps))
        textsize    = cv2.getTextSize(tmp_text, _FONT_HUD_, 1, 1)[0]
        cv2.putText(img_full, tmp_text, (1, 1 + textsize[1]), _FONT_HUD_, 1, (255, 0, 0), 1, cv2.LINE_AA)

        # Display
        cv2.imshow("img_full", img_full)
        # cv2.imshow("img_full_cvt", img_full_cvt)

        if cv2.waitKey(1) & 0xff == 27:
            print("Esc is pressed.\nExit")
            break

        t_end   = time.perf_counter()
        t_fps   = 1 / (t_end - t_begin)

        #region reduce fps

        if t_end - t_begin < (1 / 30):
            t_target    = (1 / 30)
            time.sleep(t_target - (t_end - t_begin))

            t_end   = time.perf_counter()
            t_fps   = 1 / (t_end - t_begin)

        #endregion

except KeyboardInterrupt:
    print("KeyboardInterrupt")
    pass

except Exception:
    print("Exception", sys.exc_info())
    pass
'''EDITHERE'''
# cap.release()
cv2.destroyAllWindows()

#endregion

# END
print("Closing socket...")
SOCK_CONN.close()

print("[END]")