import numpy as np
# moduly v lokálním adresáři musí být v pythonu 3 importovány s tečkou
from . import podpurne_funkce

class InstrumentTracker():
    def __init__(self):
        pass

    def predict(self, video_filename):
        """
        :param video_filename: name of the videofile
        :return: annnotations
        """
        print("ahoj")

        annotation={
            "filename": ["vid1.mp4", "vid1.mp4", "vid1.mp4"], # pth.parts[-1]
            "frame_id": [0,1,1,],
            "object_id": [0,0,1],
            "x_px": [110, 110, 300], # x pozice obarvených hrotů v pixelech
            "y_px": [50, 50, 400],   # y pozice obarvených hrotů v pixelech
            "annotation_timestamp": [],
        }
        return annotation