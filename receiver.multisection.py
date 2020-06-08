#!/usr/bin/python3

# import python modules
import torch
import torch.nn as nn
import numpy as np
import cv2

# import custom modules
from model.carn_m import Net as model
from utils.transformer import Transform,deTransform

import SpoutSDK
import time
import cv2
import math, numpy as np
import socket
import sys, os
import zlib
import threading
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
from PIL import Image
PRESET_IMG_SIZE = (int(2560 * 0.5*0.25), int(960 * 1*0.25)) #set output resolution must same as sender
PRESET_IMG_SECTION_NUM  = 16 # must same as sender

SOCK_CONN   = None
SOCKET_HOST = "0.0.0.0" # any interface
SOCKET_PORT = 1112
PROTOCOL_DATA_DELIMITER = b"[HEADER]"
SOCKET_BUFF_SIZE = int((PRESET_IMG_SIZE[0] * PRESET_IMG_SIZE[1]) * 3) + len(PROTOCOL_DATA_DELIMITER) + 1

_FONT_HUD_  = cv2.FONT_HERSHEY_SIMPLEX

def MyThreadStreamReader():

    print(">MyThreadStreamReader() => {}".format(threading.current_thread().getName()))

    global SOCK_CONN, SOCKET_PORT
    SOCKET_BUFFER_DATA  = b""

    index_begin = None
    index_end   = None

    t_begin = 0
    t_end   = 0
    t_fps   = 0

    try:

        while True:

            # read data
            data, addr  = SOCK_CONN.recvfrom(SOCKET_BUFF_SIZE)
            SOCKET_BUFFER_DATA  += data

            #region Find HEADER
            while(True):
                if index_begin == None:
                    res = SOCKET_BUFFER_DATA.find(PROTOCOL_DATA_DELIMITER)
                    if res >= 0:
                        # print("Begin index = ", res)
                        index_begin = res
                        index_end   = None
                    else:
                        break
                else:
                    res = SOCKET_BUFFER_DATA.find(PROTOCOL_DATA_DELIMITER, index_begin + len(PROTOCOL_DATA_DELIMITER))
                    if res > index_begin:
                        # print("End index = ", res)
                        index_end   = res

                        try:

                            #region Data Extraction
                            package_number = SOCKET_BUFFER_DATA[index_begin + len(PROTOCOL_DATA_DELIMITER):index_begin + len(PROTOCOL_DATA_DELIMITER) + 1]
                            img_package_num = int.from_bytes(package_number, "big")
                            if img_package_num < 0 or img_package_num >= PRESET_IMG_SECTION_NUM:
                                # Decoding fail
                                print("Skip bad header ({})".format(img_package_num))
                            else:
                                data_extract    = SOCKET_BUFFER_DATA[index_begin + len(PROTOCOL_DATA_DELIMITER) + 1:index_end]
                                # print("Receive (pkg#{}) {} byte".format(img_package_num, len(data_extract)))

                                if len(data_extract) > 0:
                                    img_recv_raw    = np.frombuffer(data_extract, np.uint8)

                                    # print("Decoding ...")
                                    # img_recv_raw_cvt    = img_recv_raw.reshape(PRESET_IMG_SIZE[1], PRESET_IMG_SIZE[0], 3)   # image raw
                                    img_recv_decode     = cv2.imdecode(img_recv_raw, cv2.IMREAD_COLOR)  # image decode

                                    if img_recv_decode is not None:
                                        
                                        # Draw
                                        # print("Performance calculation")
                                        t_end   = time.perf_counter()
                                        t_fps   = 1 / (t_end - t_begin)
                                        t_begin = time.perf_counter()

                                        tmp_text    = str(round(t_fps))
                                        textsize    = cv2.getTextSize(tmp_text, _FONT_HUD_, 1, 1)[0]
                                        # cv2.putText(img_recv_raw_cvt, tmp_text, (1, 1 + textsize[1]), _FONT_HUD_, 1, (255, 0, 0), 1, cv2.LINE_AA)    # image raw
                                        # cv2.putText(img_recv_decode, tmp_text, (1, 1 + textsize[1]), _FONT_HUD_, 1, (255, 0, 0), 1, cv2.LINE_AA)  # image decode

                                        # Recontrsuction Data
                                        # print("Reconstruction ...")
                                        img_height_per_pkg  = int(PRESET_IMG_SIZE[1] / PRESET_IMG_SECTION_NUM)
                                        img_py_start        = int(img_package_num * img_height_per_pkg)
                                        IMAGE_RESULT_DATA[img_py_start:img_py_start + img_height_per_pkg, 0:PRESET_IMG_SIZE[0]]  = img_recv_decode

                                    else:
                                        print("Skip decoding section {} fail !".format(img_package_num))

                                else:
                                    print("! Skip empty data in section {}".format(img_package_num))

                            # remove DATA from buffer (shift data to begin position)
                            # print("shift data to begin position")
                            SOCKET_BUFFER_DATA  = SOCKET_BUFFER_DATA[index_end:]

                            # reset index
                            index_begin = index_end = None

                            #endregion

                        except Exception:
                            print("! Data Extraction :", sys.exc_info())
                    else:
                        break


            # Clear Buffer
            if len(SOCKET_BUFFER_DATA) > SOCKET_BUFF_SIZE * 2:
                # clear tmp
                print("Clear buffer ({})!".format(len(SOCKET_BUFFER_DATA)))
                SOCKET_BUFFER_DATA  = b""

            #endregion

    except Exception:
        print("! MyThreadStreamReader() :", sys.exc_info())

    finally:
        print("<MyThreadStreamReader() => {}".format(threading.current_thread().getName()))

#=======================================================================================================================================#

print("[START] in {}".format(os.getcwd()))

print("Initialize socket {}:{} ...".format(SOCKET_HOST, SOCKET_PORT))
SOCK_CONN   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print("Binding ...")
SOCK_CONN.bind((SOCKET_HOST, SOCKET_PORT))
SOCK_CONN.settimeout(10)

# set socket option
bufsize = SOCK_CONN.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
print("Socket sending buffer size = ", bufsize)
bufsize = SOCK_CONN.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
print("Socket receiving buffer size = ", bufsize)

print("Waiting for data ...")


width = 800 
height = 600 
display = (width,height)

pygame.init() 
pygame.display.set_caption('Spout Python Sender')
pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
pygame.display.gl_set_attribute(pygame.GL_ALPHA_SIZE, 8)

glMatrixMode(GL_PROJECTION)
gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)
# reset the drawing perspective
glLoadIdentity()
# can disable depth buffer because we aren't dealing with multiple geometry in our scene
glDisable(GL_DEPTH_TEST)
glEnable(GL_ALPHA_TEST)
glEnable(GL_BLEND);
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
glClearColor(0.0,0.0,0.0,0.0)
glColor4f(1.0, 1.0, 1.0, 1.0);   
glTranslatef(0,0, -5)
glRotatef(25, 2, 1, 0)
spoutSender = SpoutSDK.SpoutSender()
spoutSenderWidth = width
spoutSenderHeight = height
spoutSender.CreateSender('Spout Python Sender', spoutSenderWidth, spoutSenderHeight, 0)
senderTextureID = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, 0)
glBindTexture(GL_TEXTURE_2D, senderTextureID)


try:

    IMAGE_RESULT_DATA   = np.zeros((PRESET_IMG_SIZE[1], PRESET_IMG_SIZE[0], 3), np.uint8)

    t_begin = 0
    t_end   = 0
    t_fps   = 0

    #region Create Stream Reader Thread

    thread  = threading.Thread(target=MyThreadStreamReader, name="My Stream Reader")
    thread.start()

    #endregion

    # create texture
    senderTextureID = glGenTextures(1)
    
           
    while True:
        
        try:

            glActiveTexture(GL_TEXTURE0)
            glClearColor(0.0,0.0,0.0,0.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)    
            
            # Perform a rotation and since we aren't resetting our perspective with glLoadIdentity, then each frame will perform a successive rotation on top of what we already see
            glRotatef(1, 3, 1, 1)
            glBindTexture(GL_TEXTURE_2D, senderTextureID)

            ########################
            # convert image to OpenGL texture format
            tx_image = IMAGE_RESULT_DATA
            tx_image = Image.fromarray(tx_image)     
            ix = tx_image.size[0]
            iy = tx_image.size[1]
            tx_image = tx_image.tobytes('raw', 'BGRX', 0, -1)
            glBindTexture(GL_TEXTURE_2D, senderTextureID)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, tx_image)
        ############################
            # glCopyTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,0,0,spoutSenderWidth,spoutSenderHeight,0);
            glBindTexture(GL_TEXTURE_2D, 0)

            # send texture to Spout
            # Its signature in C++ looks like this: bool SendTexture(GLuint TextureID, GLuint TextureTarget, unsigned int width, unsigned int height, bool bInvert=true, GLuint HostFBO = 0);
            spoutSender.SendTexture(int(senderTextureID), GL_TEXTURE_2D, ix, iy, True, 0)



            cv2.imshow("IMAGE_RESULT_DATA", IMAGE_RESULT_DATA)           # image decode
            '''
            '''
            
            HALFPRECISION = True # calculate with half precision or not (for a bit faster inference)

            # initialize the neural network with haft-precision inference mode and load saved parameters
            # note that call this section once when start
            neural_net = model(scale=4,group=4,multi_scale=True).cuda()
            neural_net.load_state_dict(torch.load("checkpoint/carn_m.pth"))
            neural_net.eval()
            if HALFPRECISION:
                neural_net.half()



            # downscale the image 16 times (320 x 280)
            # the image "lr" will be streamed
            lr = IMAGE_RESULT_DATA.copy()

            # inference
            # take a low resolution image "lr" of size 1280x960 as input
            # use Transform function to convert lr to tensor locating on GPU
            # yield a CUDA tensor (GPU) of one image of size 1280 x 960
            sr = neural_net(Transform(lr,halfprecision = HALFPRECISION),scale=4)

            # bring the image back to CPU
            # return as numpy array of super resoluted image of size 1280 x 960
            # this process has huge computational time, GOOD LUCK!
            sr =deTransform(sr)


            # show the result :)
            cv2.imshow('image',sr)
            


            '''
            '''
            if cv2.waitKey(1) & 0xff == 27:
                print("Esc is pressed.\nExit")

                print("Closing socket ...")
                SOCK_CONN.close()

                break

        except Exception:
            print("Exception : ", sys.exc_info())

except KeyboardInterrupt:
    print("KeyboardInterrupt")

except Exception:
    print("Exception", sys.exc_info())

print("[END]")