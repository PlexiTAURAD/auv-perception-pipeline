import numpy as np
import threading
import time

class SonarSimulator:
    def __init__(self, sensor_buffer):
        self.buffer = sensor_buffer
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.simulate_sonar_data)
        self.thread.start()
    
    def stop(self):
        self.running = False
        self.thread.join() # blocks the calling thread | ensures clean shutdown

    def simulate_sonar_data(self):
        # only get a 1d array of 100 values
        while self.running:
            sonar_data = np.random.rand(100).astype(np.float32) # Simulate sonar data as a 1D array of 100 random values
            self.buffer.push_frame(sonar_data) 
            time.sleep(0.1) # 10 Hz