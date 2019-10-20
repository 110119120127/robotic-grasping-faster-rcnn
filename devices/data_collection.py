
import pyrealsense2 as rs
import numpy as np
import cv2
import os

class RealSenseD400:
    def __init__(self):
        
        cwd = os.getcwd()
        self.image_path = cwd + '//images'

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        #self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        ctx = rs.context()
        self.cam_name = []
        if len(ctx.devices) > 0:
            for d in ctx.devices:
                self.cam_name.append(d.get_info(rs.camera_info.name))
                print ('Found device: ', \
                        d.get_info(rs.camera_info.name), ' ', \
                        d.get_info(rs.camera_info.serial_number))
        else:
            print("No Intel Device connected")

        self.k_streaming_stable = 30 # manually adjusted, make sure streaming stable  
        width_resized = 640
        height_resized = 480
        self.dim_resized = (width_resized, height_resized)
        # Start streaming
        self.pipeline.start(self.config)
        (self.x,self.y,self.w,self.h) = (275, 165, 640, 480)
    
    def start(self):
        print(self.cam_name[0] + ' starts streaming')
        k = 0
        kk = 0              # kk > k_streaming_stable --> start processing
        bRoISelected = False
        while True:
            kk += 1
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            #depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            #if not depth_frame or not color_frame:
            if not color_frame:
                continue

            # Convert images to numpy arrays
            #depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            if kk > self.k_streaming_stable:       # streaming is stable now
                '''if not bRoISelected:
                    slect_win = "Select RoI"
                    (x,y,w,h) = cv2.selectROI(slect_win, color_image, False)
                    
                    if w > 0: # wait until the user selects an RoI
                        cv2.destroyWindow(slect_win)
                        print((x,y,w,h))
                        bRoISelected = True '''
                # after manually select an RoI, now we can choose: (x,y,w,h) = (275, 165, 640, 480)
                
                bRoISelected = True
                if bRoISelected:                         
                    color_image_roi = color_image[self.y:self.y+self.h, self.x:self.x+self.w]
                    #depth_colormap = depth_colormap[x0:x1, y0:y1]
                    # Stack both images horizontally
                    #images = np.hstack((color_image, depth_colormap))
                    
                    # resize images
                    img_color_resized = cv2.resize(color_image_roi, self.dim_resized, interpolation = cv2.INTER_LINEAR)
                    images = img_color_resized

                    cv2.rectangle(color_image, (self.x, self.y), (self.x+self.w, self.y+self.h), (0,0,0), 1)
                    
                    # Show images
                    
                    cv2.namedWindow(self.cam_name[0], cv2.WINDOW_AUTOSIZE)
                    cv2.imshow(self.cam_name[0], color_image)
                    key = cv2.waitKey(30)
                    
                    if key in [27, ord('q'), ord('Q')]:      # Esc, 'q'
                        cv2.destroyAllWindows()
                        break
                    elif key in [ord('s'), ord('S')]:        # 's'
                        k += 1
                        pic_name = str(k).zfill(5) + '.jpg'
                        cv2.imwrite(os.path.join(self.image_path , pic_name), img_color_resized)

    def stop(self):
        # Stop streaming
        self.pipeline.stop()
        print(self.cam_name[0] + ' stops streaming')


if __name__ == "__main__":
    rs400 = RealSenseD400()
    rs400.start()
    rs400.stop()

