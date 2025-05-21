import os
import socket
import struct
import logging
import datetime
import threading
import xdrlib
import numpy as np
from PIL import Image

from inference import VISAIModel
from vis_anomaly_sender import AnomalySender

# set system settings
filename = f'logs/app_{datetime.datetime.now():%Y%m%d_%H%M%S}.log'
logging.basicConfig(filename=filename,
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Function to read .env file and set environment variables
def load_env_file(file_path):
    """Loads and reads env file

    Args:
        file_path (str): file name for the env file
    """
    with open(file_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# Load environment variables from .env file
load_env_file('host_ip.env')

class UDPCollector:
    def __init__(self, model_name: str, fullRes: int, num_of_lines: int, Anomalie_IpPort: tuple):
        """initializes UDP collector class

        Args:
            model_name (str): path to torch model
            fullRes (int): Full Resolution of the image
            num_of_lines (int): Number of lines stacked for the image
            Anomalie_IpPort (tuple): Tuple of (host, port) for the anomalie receiver
        """
        self.fullRes = fullRes
        self.num_of_lines = num_of_lines
        
        self.model = VISAIModel(model_name, fullRes, num_of_lines)
        self.anom_send = AnomalySender(Anomalie_IpPort[0], Anomalie_IpPort[1], fullRes)
    
    def decode_data(self, data):
        """Decoding of the byte string from the VIS Line Scanner

        Args:
            data (bytes): data as bytes object to be decoded

        Returns:
            tuple: img (line RGBA), line_id, timestamp
        """
        base_index = 68
        line_id_start = base_index - 12
        timestamp_start = line_id_end = base_index - 8
        channels_start = timestamp_end = base_index
        w_start = channels_end = base_index + 4
        spare1_start = w_end = base_index + 8
        spare1_end = base_index + 12

        line_id = struct.unpack('>I', data[line_id_start:line_id_end])[0]
        timestamp = struct.unpack('>Q', data[timestamp_start:timestamp_end])[0]
        #channels = struct.unpack('>I', data[channels_start:channels_end])[0]
        w = struct.unpack('>I', data[w_start:w_end])[0]
        #spare1 = struct.unpack('>I', data[spare1_start:spare1_end])[0]

        raw = [0] * w
        for i in range(w):
            r = data[24 + i + w * 0] & 0xFF
            g = data[24 + i + w * 1] & 0xFF
            b = data[24 + i + w * 2] & 0xFF
            a = data[24 + i + w * 3] & 0xFF
            rgba = r | (g << 8) | (b << 16) | (a << 24)
            raw[i] = rgba

        img = Image.new("RGBA", (w, 1))
        img.putdata([(rgba >> 16 & 0xFF, rgba >> 8 & 0xFF, rgba & 0xFF) for rgba in raw])        

        return img, line_id, timestamp

    def collect_udp_data(self):
        """Collect udp data from socket
        """
        imgBuffer = np.zeros((self.fullRes,self.num_of_lines,3))
        alpha = np.zeros((self.fullRes,self.num_of_lines,1))
        i = 0
        line_ids = []
        
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.bind(('localhost', 7799))  # Replace with your UDP port

        while True:
            xdr_data, _ = udp_socket.recvfrom(18512)
            if len(xdr_data) < 10000:
                # skip description packages 
                continue

            image, line_id, timestamp = self.decode_data(xdr_data)
            image = np.array(image)
            line_ids.append(line_id)

            # stack lines
            image = np.squeeze(image)
            imgBuffer[:,i,:] = image[700:3900,:3] # rgb image from rgba
            alpha[:,i] = image[700:3900,3].reshape(-1, 1) # alpha channel from rgba
            i += 1
            
            # Check if we have collected the required number of lines
            if i == self.num_of_lines:
                # Process the collected data
                threading.Thread(target=self.evaluate_data,
                                 args=(imgBuffer.copy(), alpha.copy(), line_ids.copy()),
                                 daemon=True).start()
                
                # Reset the counter and clear the data queue
                i = 0
                imgBuffer.fill(0)
                alpha.fill(0)
                line_ids.clear()

    def evaluate_data(self, imgBuffer: np.ndarray, alpha: np.ndarray, line_ids: list):
        """Starts the model inference and calls the anomalie sender

        Args:
            imgBuffer (np.ndarray): RGB image for inference
            alpha (np.ndarray): Alpha channel containing flag information
            line_ids (list): list of lineIDs
        """
        try:
            size, medoids_list = self.model.inference(imgBuffer, alpha)
        except Exception as e:
            size = 0
            logging.error("Exception occurred", exc_info=True)
        
        # no hotspots detected
        if size < 1:
            logging.info('No anomalies found')
        else:
            # send data to com
            logging.info(f'Anomalies found {size}')
            self.anom_send.get_anomalies_from_medoids(medoids_list, line_ids)


if __name__=='__main__':
    # Load environment variables from host_ip.env file
    fullRes = 3200 #TODO:/FIXME: New models and resolutions if applicable
    num_of_lines = 480 #TODO:/FIXME: New models and resolutions if applicable

    hostname = os.getenv('HOSTNAME')
    local_port = int(os.getenv('LOCAL_PORT'))
    unique_view_ID = int(os.getenv('UNIQUE_VIEW_ID'))
    view_channel = os.getenv('VIEW_CHANNEL')
    model_name = os.getenv('MODEL_NAME')

    # set up resolution, models and stuff #TODO:/FIXME: New models and resolutions if applicable
    model_name = f'./models/{model_name}'
    fullRes = 3200 #TODO:/FIXME: New models and resolutions if applicable
    num_of_lines = 480 #TODO:/FIXME: New models and resolutions if applicable

    local_IpPort = (hostname, local_port)
    VIS_IpPort = (os.getenv('VIS_IP'), int(os.getenv('VIS_PORT')))
    Anomalie_IpPort = (os.getenv('ANOMALIE_IP'), int(os.getenv('ANOMALIE_PORT')))

    # define ip:port home/ receiver
    UDP_rec = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDP_rec.bind(local_IpPort)
    
    # send message to register receiver in VIS line scanner
    p = xdrlib.Packer()
    p.pack_int(unique_view_ID) # unique view ID
    p.pack_string(hostname.encode()) # my ip
    p.pack_int(local_port) # my port to receive VIS data
    p.pack_string(view_channel.encode())

    signup_msg = p.get_buffer()

    UDP_rec.sendto(signup_msg, VIS_IpPort)
    print("Tach sent", signup_msg)
    UDP_rec.close()

    collector = UDPCollector(model_name, fullRes, num_of_lines, Anomalie_IpPort)
    collector.collect_udp_data()
