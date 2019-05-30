import os

class CKdata():
    def __init__(self, data_dir=None):
        self.imgs_path = os.path.join(data_dir, 'cohn-kanade-images')
        self.Emotion_path = os.path.join(data_dir, 'Emotion')
        self.FACS_path = os.path.join(data_dir, 'FACS')
        self.Landmarks_path = os.path.join(data_dir, 'Landmarks')

