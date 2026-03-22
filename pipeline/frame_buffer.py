from collections import deque
import threading

class FrameBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=1) # Initialize a deque with a maximum length of 1
        self.new_frame_event = threading.Event() 

    def push_frame(self, frame):
        self.buffer.append(frame)
        self.new_frame_event.set() # Signal that a new frame is available
        
    def get_frame(self):
        self.new_frame_event.wait()
        self.new_frame_event.clear() # Right after this line, the yolo thread should run and force the GIL for it's code
        return self.buffer[-1] # Return the most recent frame, can also be 0 since maxlen=1 but this is good practice I believe
    
    