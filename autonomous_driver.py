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
        
        # --- THE AMPLIFIER ---
        # The AI suffers from "Regression to the Mean" (whispering its steering).
        # We multiply it by 100 to force the servo out of its deadzone!
        self.STEERING_GAIN = 100.0 
        
        print("[AI] Optimizing ONNX for Raspberry Pi 5 (4 Cores)...")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4 
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
        if rclpy.ok():
            try:
                self.cmd_pub.publish(twist)
            except Exception:
                pass

    def drive_loop(self):
        print("\n[SYSTEM] Waiting for camera feed...")
        while self.latest_image is None and self.is_driving:
            time.sleep(0.1)
            
        print("[SYSTEM] Camera detected! Starting Autonomous AI Loop...")
        
        # Slowed down slightly so the 0.7 FPS AI has more time to see the curves
        CRUISE_SPEED = 0.08 
        
        while self.is_driving:
            start_time = time.monotonic()
            
            with self.lock:
                img_copy = self.latest_image.copy()
                
            tensor_img = self.preprocess_image(img_copy)
            
            latent_vector = self.encoder_session.run(None, {self.encoder_input_name: tensor_img})[0]
            action = self.actor_session.run(None, {self.actor_input_name: latent_vector})[0]
            
            raw_steer = float(action[0][0])
            steering = raw_steer * self.STEERING_GAIN
            throttle = CRUISE_SPEED
            
            # HARD CAP: Prevent the massive gain from breaking the physical servo
            steering = max(min(steering, 1.0), -1.0)
            
            twist = Twist()
            twist.angular.z = steering
            twist.linear.x = throttle
            
            if self.is_driving:
                self.safe_publish(twist)
            
            process_time = time.monotonic() - start_time
            fps = 1.0 / process_time if process_time > 0 else 0.0
            print(f"AI Steer: {steering:+.3f} (Raw: {raw_steer:+.3f}) | Throttle: {throttle:+.3f} | {fps:.1f} FPS")

    def emergency_stop(self):
        print("\n[SYSTEM] EMERGENCY STOP. Waiting for AI thread to finish...")
        self.is_driving = False 
        
        if self.ai_thread.is_alive():
            self.ai_thread.join() 
            
        print("[SYSTEM] AI Halted. Sending zero velocity to motors...")
        twist = Twist()
        twist.angular.z = 0.0
        twist.linear.x = 0.0
        
        if rclpy.ok():
            try:
                # Send the stop command
                self.cmd_pub.publish(twist)
                print("[SYSTEM] Stop command sent...")
                
                # CRITICAL FIX: Sleep for 0.5s to let the network buffer actually deliver the message!
                time.sleep(0.5) 
                
                # Send it one more time just in case
                self.cmd_pub.publish(twist)
                time.sleep(0.2)
                
                print("[SYSTEM] Motors successfully halted.")
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
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()