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

class SynchronizedDataCollector(Node):
    def __init__(self):
        super().__init__('sync_data_collector')
        
        # 1. Configuration
        self.frequency = 15.0
        self.time_step = 1.0 / self.frequency
        self.max_steps = 10000  # Records up to ~11 minutes
        self.is_recording = True
        self.step_count = 0
        
        self.bridge = CvBridge()
        
        # 2. Shared Memory Variables & Lock
        self.lock = threading.Lock()
        
        # Default empty arrays
        self.latest_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.latest_imu = np.zeros(6, dtype=np.float32)
        self.latest_action = np.zeros(2, dtype=np.float32)
        
        # 3. ROS 2 Subscribers
        self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.img_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        
        # 4. Start the strict hardware loop in a background thread
        self.record_thread = threading.Thread(target=self.strict_record_loop)
        self.record_thread.start()
        
        self.get_logger().info("ROS 2 Node Started. Waiting for sensor data...")

    # --- ROS 2 CALLBACKS (Updating Shared Memory) ---
    
    def img_callback(self, msg):
        try:
            # Convert to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Force size to exactly 640x480 to prevent HDF5 shape crashes
            frame = cv2.resize(frame, (640, 480))
            
            with self.lock:
                self.latest_image = frame
        except Exception as e:
            self.get_logger().error(f"Image Error: {e}")

    def imu_callback(self, msg):
        with self.lock:
            # Save: [Accel X, Accel Y, Accel Z, Gyro X, Gyro Y, Gyro Z]
            self.latest_imu = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ], dtype=np.float32)

    def cmd_callback(self, msg):
        with self.lock:
            # Save: [Steering (Angular Z), Throttle (Linear X)]
            self.latest_action = np.array([
                msg.angular.z, 
                msg.linear.x
            ], dtype=np.float32)

    # --- STRICT TIMING LOOP (Saving to HDF5) ---
    
    def strict_record_loop(self):
        print("\n[RECORDER] Initializing HDF5 file 'driving_dataset_001.h5'...")
        file = h5py.File("driving_dataset_001.h5", "w")
        
        # Pre-allocate memory (stops CPU freezing)
        dataset_image = file.create_dataset("observations_image", (self.max_steps, 480, 640, 3), dtype='uint8')
        dataset_imu = file.create_dataset("observations_imu", (self.max_steps, 6), dtype='float32')
        dataset_action = file.create_dataset("actions", (self.max_steps, 2), dtype='float32')
        
        print(f"[RECORDER] Ready! Recording at exactly {self.frequency} Hz.")
        print("[RECORDER] Grab your Gamepad and DRIVE! Press Ctrl+C to stop.")
        
        next_loop_time = time.monotonic()
        
        while self.is_recording and self.step_count < self.max_steps:
            # 1. Grab copies of the data safely
            with self.lock:
                img_copy = self.latest_image.copy()
                imu_copy = self.latest_imu.copy()
                act_copy = self.latest_action.copy()
                
            # 2. Write to file
            dataset_image[self.step_count] = img_copy
            dataset_imu[self.step_count] = imu_copy
            dataset_action[self.step_count] = act_copy
            
            self.step_count += 1
            
            # 3. Calculate exact sleep duration
            next_loop_time += self.time_step
            sleep_duration = next_loop_time - time.monotonic()
            
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                # If negative, we missed our deadline
                print(f"[WARNING] Frame drop at step {self.step_count}! Loop too slow.")
                # Reset the clock so we don't try to catch up infinitely
                next_loop_time = time.monotonic()
                
        # --- SHUTDOWN & CLEANUP ---
        print(f"\n[RECORDER] Stopping... Resizing datasets to {self.step_count} steps.")
        dataset_image.resize((self.step_count, 480, 640, 3))
        dataset_imu.resize((self.step_count, 6))
        dataset_action.resize((self.step_count, 2))
        
        file.close()
        print("[RECORDER] Dataset saved successfully! You can now transfer 'driving_dataset_001.h5' to your Mac.")

def main(args=None):
    rclpy.init(args=args)
    node = SynchronizedDataCollector()
    
    try:
        # This keeps the ROS 2 callbacks running
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nCtrl+C detected! Shutting down...")
    finally:
        # Signal the recording thread to stop
        node.is_recording = False
        node.record_thread.join() # Wait for the HDF5 file to close safely
        
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()