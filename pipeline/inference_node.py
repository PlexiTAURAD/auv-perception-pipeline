import onnxruntime as ort
import numpy as np
import threading
import cv2

# model path mentioned in main.py

class InferenceNode:
    def __init__(self, model_path):
        #self.frame_buffer = frame_buffer
        self.model_path = model_path
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, frame):
        # Convert (H, W, 3) to (1, 3, H, W)
        frame = np.transpose(frame, (2, 0, 1))[None, :] # 2 -> "3" , 0 -> "H" , 1 -> "W" , [None, :] adds a new axis at the beginning for batch size
        # Convert frame to fp32 and normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def process_frame(self,frame):
        #frame = self.frame_buffer.get_frame()
        preprocessed_frame = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: preprocessed_frame}) # {input_feed : Any}
        #print("Output shape:", [output.shape for output in outputs])
        #print("Inference output:", outputs)
        return self.postprocess(outputs)

    def postprocess(self, outputs, conf_threshold=0.8, iou_threshold=0.5):
        # Outputs are in onnx format (1x84x8400)
        predictions = outputs[0][0] # (84x8400)
        boxes = predictions[:4, :] # (4x8400)
        scores = predictions[4:, :] # (80x8400)
        
        best_scores = np.max(scores, axis=0) # (8400,)
        best_classes = np.argmax(scores, axis=0) # (8400,)

        mask = best_scores > conf_threshold
        filtered_boxes = boxes[:, mask]
        filtered_scores = best_scores[mask]
        filtered_classes = best_classes[mask]

        # need this because of np.dnn.NMSBoxes
        x_min = filtered_boxes[0, :] - filtered_boxes[2, :] / 2
        y_min = filtered_boxes[1, :] - filtered_boxes[3, :] / 2
        width = filtered_boxes[2, :]
        height = filtered_boxes[3, :]

        nms_boxes = np.column_stack((x_min, y_min, width, height)).tolist() 
        nms_scores = filtered_scores.tolist()

        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, conf_threshold, iou_threshold)
        detections = []

        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    # getting X_center
                    'bbox': [
                        float(x_min[i]),
                        float(y_min[i]),
                        float(x_min[i] + width[i]),
                        float(y_min[i] + height[i])
                    ],
                    'confidence': float(filtered_scores[i]),
                    'class': int(filtered_classes[i]),
                    'x_center_norm' : float(filtered_boxes[0, i] / 640),
                
                })
        return detections

        