import time
from pipeline.camera_source import CameraSource
from pipeline.frame_buffer import FrameBuffer
from pipeline.inference_node import InferenceNode
from pipeline.sonar_simulator import SonarSimulator 

def main():
    camera_buffer = FrameBuffer()
    sonar_buffer = FrameBuffer()

    node = InferenceNode("model/yolov8n.onnx") 

    camera_source = CameraSource(camera_buffer)
    sonar_source = SonarSimulator(sonar_buffer)

    camera_source.start_pipeline()
    sonar_source.start()

    try:
        while True:
            camera_frame = camera_buffer.get_frame() 
            sonar_array = sonar_buffer.get_frame()   

            detections = node.process_frame(camera_frame) 

            for det in detections:
                class_id = det['class']
                confidence = det['confidence']
                x_norm = det['x_center_norm']

                # Index will be calculated as: min(int(x_norm x 100),99)
                sonar_index = min(int(x_norm * 100), 99)
                distance = sonar_array[sonar_index] 

                print(f"Fused Object -> Class: {class_id} | Conf: {confidence:.2f} | Distance: {distance:.2f}m")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down gracefully...")
    finally:
        # Cleanly sever hardware locks and OS threads
        camera_source.stop_pipeline()
        sonar_source.stop() # Or whatever your stop method is named

if __name__ == "__main__":
    main()