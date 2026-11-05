import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import cv2
import numpy as np
import threading
import time
import onnxruntime as ort

class AutonomousDriver(Node):
    def __init__(self):
        super().__init__('autonomous_driver')
        
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_image = None
        self.is_driving = True
        
        # --- AI GAIN MULTIPLIERS ---
        # If the AI outputs 0.05, it multiplies by 3.0 to output 0.15 (Overcoming the deadzone)
        self.STEERING_GAIN = 3.0
        self.THROTTLE_GAIN = 4.0
        
        print("[AI] Optimizing ONNX for Raspberry Pi 5 (4 Cores)...")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4  # Use all 4 Pi 5 cores!
        opts.inter_op_num_threads = 1
        
        self.encoder_session = ort.InferenceSession('dinov2_encoder.onnx', sess_options=opts, providers=['CPUExecutionProvider'])
        self.actor_session = ort.InferenceSession('actor_policy.onnx', sess_options=opts, providers=['CPUExecutionProvider'])
        
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.actor_input_name = self.actor_session.get_inputs()[0].name
        print("[AI] Brains loaded successfully!")
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.img_callback, 10)
        
        self.ai_thread = threading.Thread(target=self.drive_loop)
        self.ai_thread.start()
        
    def img_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.resize(frame, (640, 480))
            with self.lock:
                self.latest_image = frame
        except Exception:
            pass
            
    def preprocess_image(self, frame):
        cropped = frame[:, 80:560]
        resized = cv2.resize(cropped, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        img_float = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (img_float - mean) / std
        
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def safe_publish(self, twist):
        # Prevents the Ctrl+C crash by checking if ROS is still alive
        if rclpy.ok() and self.is_driving:
            try:
                self.cmd_pub.publish(twist)
            except Exception:
                pass

    def drive_loop(self):
        print("\n[SYSTEM] Waiting for camera feed...")
        while self.latest_image is None and self.is_driving:
            time.sleep(0.1)
            
        print("[SYSTEM] Camera detected! Starting Autonomous AI Loop...")
        
        while self.is_driving:
            start_time = time.monotonic()
            
            with self.lock:
                img_copy = self.latest_image.copy()
                
            tensor_img = self.preprocess_image(img_copy)
            
            latent_vector = self.encoder_session.run(None, {self.encoder_input_name: tensor_img})[0]
            action = self.actor_session.run(None, {self.actor_input_name: latent_vector})[0]
            
            # Apply the Multipliers!
            raw_steer = float(action[0][0])
            raw_throttle = float(action[0][1])
            
            steering = raw_steer * self.STEERING_GAIN
            throttle = raw_throttle * self.THROTTLE_GAIN
            
            twist = Twist()
            twist.angular.z = steering
            twist.linear.x = throttle
            self.safe_publish(twist)
            
            process_time = time.monotonic() - start_time
            fps = 1.0 / process_time if process_time > 0 else 0.0
            print(f"AI Steer: {steering:+.3f} | Throttle: {throttle:+.3f} | Speed: {fps:.1f} FPS")

    def emergency_stop(self):
        print("\n[SYSTEM] EMERGENCY STOP. Halting motors...")
        self.is_driving = False
        twist = Twist()
        twist.angular.z = 0.0
        twist.linear.x = 0.0
        if rclpy.ok():
            try:
                self.cmd_pub.publish(twist)
            except:
                pass

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousDriver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.emergency_stop()
    finally:
        node.is_driving = False
        node.ai_thread.join()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()