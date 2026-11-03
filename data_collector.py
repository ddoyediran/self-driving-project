import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import cv2
import numpy as np
import h5py
import time
import threading
import queue

class SynchronizedDataCollector(Node):
    def __init__(self):
        super().__init__('sync_data_collector')
        
        self.frequency = 15.0
        self.time_step = 1.0 / self.frequency
        self.max_steps = 10000 
        self.is_recording = True
        
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        # Shared Memory Variables
        self.latest_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.latest_imu = np.zeros(6, dtype=np.float32)
        self.latest_action = np.zeros(2, dtype=np.float32)
        
        # The Holding Queue (Absorbs the SD Card delay)
        self.data_queue = queue.Queue()
        
        # ROS 2 Subscribers
        self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.img_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        
        # Start the Background Disk Writer Thread
        self.writer_thread = threading.Thread(target=self.disk_writer_loop)
        self.writer_thread.start()
        
        # Start the Strict Timer Thread
        self.record_thread = threading.Thread(target=self.strict_record_loop)
        self.record_thread.start()
        
        self.get_logger().info("ROS 2 Node Started. Waiting for sensor data...")

    def img_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.resize(frame, (640, 480))
            with self.lock:
                self.latest_image = frame
        except Exception as e:
            pass

    def imu_callback(self, msg):
        with self.lock:
            self.latest_imu = np.array([
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
            ], dtype=np.float32)

    def cmd_callback(self, msg):
        with self.lock:
            self.latest_action = np.array([msg.angular.z, msg.linear.x], dtype=np.float32)

    # --- THREAD 1: The Strict Timer (Runs at exactly 15Hz) ---
    def strict_record_loop(self):
        print("\n[TIMER] Ready! Grab your Gamepad and DRIVE! Press Ctrl+C to stop.")
        
        step_count = 0
        next_loop_time = time.monotonic()
        
        while self.is_recording and step_count < self.max_steps:
            # 1. Grab copies safely and instantly
            with self.lock:
                img_copy = self.latest_image.copy()
                imu_copy = self.latest_imu.copy()
                act_copy = self.latest_action.copy()
                
            # 2. Throw into the queue (Takes 0.0001 seconds)
            self.data_queue.put((step_count, img_copy, imu_copy, act_copy))
            step_count += 1
            
            # 3. Calculate sleep
            next_loop_time += self.time_step
            sleep_duration = next_loop_time - time.monotonic()
            
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                print(f"[WARNING] Timer hiccup at step {step_count}! CPU is busy.")
                next_loop_time = time.monotonic()
                
        print(f"\n[TIMER] Finished timing {step_count} frames. Waiting for SD card to finish saving...")

    # --- THREAD 2: The Disk Writer (Runs as fast as the SD card allows) ---
    def disk_writer_loop(self):
        print("[WRITER] Initializing HDF5 file 'driving_dataset_001.h5'...")
        file = h5py.File("driving_dataset_001.h5", "w")
        
        dataset_image = file.create_dataset("observations_image", (self.max_steps, 480, 640, 3), maxshape=(None, 480, 640, 3), dtype='uint8')
        dataset_imu = file.create_dataset("observations_imu", (self.max_steps, 6), maxshape=(None, 6), dtype='float32')
        dataset_action = file.create_dataset("actions", (self.max_steps, 2), maxshape=(None, 2), dtype='float32')
        
        last_step = 0
        
        # Keep writing as long as we are recording, OR if there is still data stuck in the queue
        while self.is_recording or not self.data_queue.empty():
            try:
                # Pull the next frame out of the queue
                step, img, imu, act = self.data_queue.get(timeout=0.1)
                
                # Write it to the SD card
                dataset_image[step] = img
                dataset_imu[step] = imu
                dataset_action[step] = act
                
                last_step = step + 1
            except queue.Empty:
                pass # Normal behavior, just wait for more data
                
        print(f"[WRITER] All data written! Trimming file to {last_step} steps...")
        dataset_image.resize((last_step, 480, 640, 3))
        dataset_imu.resize((last_step, 6))
        dataset_action.resize((last_step, 2))
        
        file.close()
        print("[WRITER] Dataset saved successfully! You can now transfer 'driving_dataset_001.h5' to your Mac.")

def main(args=None):
    rclpy.init(args=args)
    node = SynchronizedDataCollector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nCtrl+C detected! Shutting down cleanly...")
    finally:
        node.is_recording = False
        node.record_thread.join()
        node.writer_thread.join() # Wait for both threads to finish
        
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()