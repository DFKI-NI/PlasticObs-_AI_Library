# libraries
import socket
import struct
import numpy as np

import logging


# Anomaly Class
class Anomaly:
    """Class to build anomalie objects
    """
    def __init__(self, id, line_id, pixel_id, priority, probability) -> None:
        """_summary_

        Args:
            id (int): _description_
            line_id (int): _description_
            pixel_id (int): _description_
            priority (int): _description_
            probability (float): _description_
        """
        self.id = id
        self.line_id = line_id
        self.pixel_id = pixel_id
        self.priority = priority
        self.probability = probability

# Sending Anomalies
class AnomalySender:
    """Classs to send anomalies
    """
    def __init__(self, receiver_host: str, receiver_port: int, fullRes: int = 3200) -> None:
        """Initalize anomaly sender

        Args:
            receiver_host (str): Hostname to where the data should be send
            receiver_port (int): port to where the data should be send
            fullRes (int, optional): Number of lines from the VAE model. Defaults to 3200.
        """
        self.anomalie_counter = 0
        self.receiver_host = receiver_host
        self.receiver_port = receiver_port
        self.fullRes = fullRes

    def _send_anomalies(self, anomalies) -> None:
        """Sends the anomalies to the anomaly receiver

        Args:
            anomalies (object): Instance of Anomaly class
        """
        size = 4 + (5 * 4) * len(anomalies)  # num + num * anomalies in bytes
        baos = bytearray(size)
        offset = 4

        # Pack the number of elements
        struct.pack_into(">I", baos, 0, len(anomalies))

        # Pack each anomaly
        for ano in anomalies:
            # Adjust the format string to include all 5 items
            struct.pack_into(">IIIIf", baos, offset, ano.id, ano.line_id, ano.pixel_id, ano.priority, ano.probability)
            offset += 20  # 5 integers * 4 bytes each

            #logging.info(ano.id, ano.line_id, ano.pixel_id, ano.priority, ano.probability)

        # Send the datagram
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.sendto(baos, (self.receiver_host, self.receiver_port))
        except Exception as e:
            logging.error(f"Failed to send anomalies: {e}")

            
    def get_anomalies_from_medoids(self, medoids: np.ndarray, line_ids: list) -> None:
        """handles the the conversion from medoids list to Anomalies

        Args:
            medoids (list): pixel-value, x-value, y-value 
            line_ids (list): _description_
        """
        anomalies = []
        for medoid in medoids:
            id = self.anomalie_counter
            self.anomalie_counter += 1
            
            line_id = int(line_ids[int(medoid[1])])
            pixel_id = int(medoid[2])
            
            probability = medoid[0]
            priority = self.get_prio(pixel_id, probability)
            
            anomalies.append(Anomaly(id=id, line_id=line_id, pixel_id=pixel_id, priority=priority, probability=probability))
            
        self._send_anomalies(anomalies)

    def get_prio(self, pixel_id, probability):
        """Calculate priority (0 is the highest) based on the position and probability

        Args:
            pixel_id (int): _description_
            probability (float): _description_

        Returns:
            int: prio (0 is the highest)
        """
        dist = -((1)/((self.fullRes//2)**(4)))*(pixel_id-self.fullRes//2)**(4)+1
        prio = (((1)/(dist))*((1)/(probability))-1)*10
        return round(prio)


if __name__=='__main__':
    VIS_IpPort = ("localhost", 7789) # 7777, Correct Port for VIS Server
    anom_send = AnomalySender(VIS_IpPort[0], VIS_IpPort[1])
    #np.array y,x
    #medoids x, y
    medoids = np.array([[1., 1. , 384.],
                        [0.99, 17.,  637.],
                        [0.99, 16.,  417.],
                        [0.99,  3.,  605.],
                        [0.9,  0.,  397.],
                        [0.96,  4.,  463.],
                        [0.8,  1.,  431.],
                        [0.7, 15.,  433.],
                        [0.5, 17.,  444.],
                        [0.6, 15.,  384.],
                        [0.97,  3.,  528.],
                        [0.97, 20.,  588.]])
    
    line_ids = [x for x in range(800)]
    print(len(line_ids))
    anom_send.get_anomalies_from_medoids(medoids, line_ids)
