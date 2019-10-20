
import pyrealsense2 as rs
import numpy as np
import os



class RealSenseD400:
    def __init__(self):
        
        self.image = np.zeros((640,480,3))
        self.color_image = np.zeros((1280,720,3))
        
        #cwd = os.getcwd()
        #self.image_path = cwd + '//images'

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
    
    def start_stream(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        #depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if color_frame:
            #if not depth_frame or not color_frame:
            # Convert images to numpy arrays
            #depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            self.color_image = color_image
            color_image_roi = color_image[self.y:self.y+self.h, self.x:self.x+self.w]
            self.image = color_image_roi
        return self.image

    
    def stop(self):
        # Stop streaming
        self.pipeline.stop()
        print(self.cam_name[0] + ' stops streaming')
        
