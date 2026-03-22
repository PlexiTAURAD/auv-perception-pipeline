import gi # GObject Introspection, used to access C libraries using Python
gi.require_version('Gst', '1.0') # Tells python which version of the C library to use
from gi.repository import Gst # Imports the GStreamer library
import numpy as np


class CameraSource:
    def __init__(self, frame_buffer):
        self.frame_buffer = frame_buffer
        self.pipeline = None
        Gst.init(None) # Initializes the GStreamer library

    def start_pipeline(self):
        ### change src to some saved video later ###
        pipeline_str = "filesrc location=data/test_video.mp4 ! decodebin ! videoscale ! videorate ! videoconvert ! video/x-raw,format=RGB,width=640,height=640,framerate=30/1 ! appsink name=mysink max-buffers=1 drop=true emit-signals=true"        
        self.pipeline = Gst.parse_launch(pipeline_str) # creates pipeline
        self.sink = self.pipeline.get_by_name("mysink") # gets appsink element
        self.sink.connect("new-sample", self.on_new_sample) # connects new sample to our on_new_sample function
        self.pipeline.set_state(Gst.State.PLAYING) # pipeline starts

    def stop_pipeline(self):
        #if hasattr(self, "pipeline") and self.pipeline is not None: # is this line too much? 
        if self.pipeline is not None: # Check if the pipeline has been initialized before trying to stop it 
            self.pipeline.set_state(Gst.State.NULL) # pipeline stops

    def on_new_sample(self, sink): 
        sample = sink.emit("pull-sample") # Pulls a sample from the sink
        if not sample:
            return Gst.FlowReturn.ERROR # If no sample is available, return an error
        caps = sample.get_caps() # Gets the capabilities of the sample
        buffer = sample.get_buffer() # Gets the buffer from the sample

        sam_struct = caps.get_structure(0) # Gets the first structure from the capabilities
        width = sam_struct.get_value('width') # Gets the width of the frame
        height = sam_struct.get_value('height') # Gets the height of the frame

        success, map_info = buffer.map(Gst.MapFlags.READ) # Maps the buffer for reading
        if not success:
            return Gst.FlowReturn.ERROR # If mapping fails, return an error
        
        try:
            image_array = np.ndarray(
                shape=(height, width, 3), 
                dtype=np.uint8, 
                buffer=map_info.data # The data from the mapped buffer
            )
            self.frame_buffer.push_frame(image_array) #### Make sure you define this ASAP

        finally:
            buffer.unmap(map_info) 

        return Gst.FlowReturn.OK # If everything is successful, return OK