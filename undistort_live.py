import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

class UndistortViewer(Node):
    def __init__(self):
        super().__init__('undistort_viewer')
        
        if not os.path.exists('fisheye_calib.npz'):
            print("Error: fisheye_calib.npz not found! Run calibrate.py first.")
            exit()
            
        with np.load('fisheye_calib.npz') as X:
            self.K, self.D, self.DIM = [X[i] for i in ('K', 'D', 'img_size')]
            
        # --- THE FIX IS HERE ---
        # balance=0.0 means crop all the crazy stretched pixels (zooms in).
        # balance=1.0 means keep everything (which creates the funhouse mirror).
        # Try 0.0, 0.5, or 0.8 to see what looks best for your track!
        balance = 0.5 
        
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, self.DIM, np.eye(3), balance=balance)
            
        # Use new_K instead of self.K for the mapping
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), new_K, self.DIM, cv2.CV_16SC2)

        self.subscription = self.create_subscription(
            Image,
            '/ascamera/camera_publisher/rgb0/image',
            self.image_callback,
            10)
            
        self.bridge = CvBridge()
        print("Starting live un-warped feed. Press 'q' to quit.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            return

        # Apply the mathematical map to flatten the frame
        undistorted_frame = cv2.remap(frame, self.map1, self.map2, 
                                      interpolation=cv2.INTER_LINEAR, 
                                      borderMode=cv2.BORDER_CONSTANT)

        # Show both side-by-side (Resized to fit on screen)
        half_raw = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        half_flat = cv2.resize(undistorted_frame, (0,0), fx=0.5, fy=0.5)
        combined = np.hstack((half_raw, half_flat))

        cv2.imshow('Left: Raw Fisheye | Right: Undistorted', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = UndistortViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()