import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import cv2
import numpy as np
import time
import threading
import queue
import csv

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
        
        # The Holding Queue
        self.data_queue = queue.Queue()
        
        # ROS 2 Subscribers
        self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.img_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        
        # Start Background Writer Thread
        self.writer_thread = threading.Thread(target=self.disk_writer_loop)
        self.writer_thread.start()
        
        # Start Strict Timer Thread
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
            with self.lock:
                img_copy = self.latest_image.copy()
                imu_copy = self.latest_imu.copy()
                act_copy = self.latest_action.copy()
                
            self.data_queue.put((step_count, img_copy, imu_copy, act_copy))
            step_count += 1
            
            next_loop_time += self.time_step
            sleep_duration = next_loop_time - time.monotonic()
            
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                print(f"[WARNING] Timer hiccup at step {step_count}! CPU is busy.")
                next_loop_time = time.monotonic()
                
        print(f"\n[TIMER] Finished timing {step_count} frames. Waiting for background compression to finish...")

    # --- THREAD 2: The Disk Writer (Encodes video and CSV in the background) ---
    def disk_writer_loop(self):
        print("[WRITER] Initializing Video and CSV files...")
        
        # Setup Video Writer (MJPG is fast and widely supported)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_out = cv2.VideoWriter('driving_dataset_001.avi', fourcc, self.frequency, (640, 480))
        
        # Setup CSV Writer
        csv_file = open('driving_dataset_001.csv', 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_id', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'steering', 'throttle'])
        
        last_step = 0
        
        while self.is_recording or not self.data_queue.empty():
            try:
                step, img, imu, act = self.data_queue.get(timeout=0.1)
                
                # Write frame to AVI
                video_out.write(img)
                
                # Write data to CSV
                csv_writer.writerow([
                    step, 
                    imu[0], imu[1], imu[2], 
                    imu[3], imu[4], imu[5], 
                    act[0], act[1]
                ])
                
                last_step = step + 1
            except queue.Empty:
                pass
                
        print(f"[WRITER] All {last_step} frames compressed and written!")
        video_out.release()
        csv_file.close()
        print("[WRITER] Success! You can now transfer 'driving_dataset_001.avi' and 'driving_dataset_001.csv' to your Mac.")

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
        node.writer_thread.join() 
        
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()