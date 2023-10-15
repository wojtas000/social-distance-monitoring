import cv2
import os 

"""
Two classes have been defined in this module in order to facilitate operation on objects storing
imported video or photo files.

"""


class VideoImport:
    """
    parametres:
        path (string) - path to the file directory
        name (string) - name of the file in the directory (optional), 
            if not specified, the first file in the directory would be imported
    """
    
    def __init__(self, path, name):
                
        if os.path.isdir(path):
            self.path = path
            if name[-4:] == '.avi':
                self.name = name
            elif type(name) != str:
                raise Exception("Wrong name of the file. It has to be given as string")
            elif name == '':
                files = os.listdir(path)
                print(f"Importing {len(files)} files")
                for file_name in files:
                    self.name = ''
                    if file_name[-4:] == '.avi':
                        self.name = file_name
                        break
                if self.name == '':
                    raise Exception("There is no file with the extension .avi in the given directory")
                
            elif name[-4:] != '.avi':
                raise Exception("Wrong name of the file. The extension has to .avi")

        elif path[-4:] == '.avi':
            self.path = '\\'.join(path.split("\\")[:-1])
            self.name = path.split("\\")[-1]
        else:
            raise Exception("Wrong path")
        
        self.video = cv2.VideoCapture(self.path+'\\'+self.name)
        self.frames_number = self.video.get(cv2.CAP_PROP_FRAME_COUNT)

    def make_frames(self, frames_number):
        """
        method creates the frames (number specified as an argument) from the video and 
        returns tensors in the list
        """
        frames = []
        for i in range(frames_number):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, i)
            is_frame, frame = self.video.read()
            if not is_frame:
                break 
            else:
                frames.append(frame)
        return frames
    

class ImageImport:
    """
    Class created for the import and operation on images given as the input for the model
    """
    
    def __init__(self, path, name=''):
        """
        parametres:
            path (string) - path to the file directory
            name (string) - name of the file in the directory (optional), 
                if not specified, the first file in the directory would be imported
        """
                
        if os.path.isdir(path):
            self.path = path
            if name[-4:] == '.jpg' or name[-4:] == '.png':
                self.name = name
            elif type(name) != str:
                raise Exception("Wrong name of the file. It has to be given as string")
            elif name == '':
                files = os.listdir(path)
                print(f"Importing {len(files)} files")
                self.name = []
                for file_name in files:
                    if file_name[-4:] == '.jpg' or file_name[-4:] == '.png':
                        self.name.append(file_name)
                if not self.name:
                    raise Exception("There is no file with the extension .jpg or .png in the given directory")
                
            if name[-4:] == '.jpg' or name[-4:] == '.png':
                raise Exception("Wrong name of the file. The extension has to .jpg or .png")

        elif path[-4:] == '.jpg' or path[-4:] == '.png':
            self.path = '\\'.join(path.split("\\")[:-1])
            self.name = path.split("\\")[-1]
        else:
            raise Exception("Wrong path")
        
        self.image = []
        
        if type(self.name) == list:
            for i in self.name:
                self.image.append(cv2.imread(self.path + "\\" + i))
        else:
            self.image.append(cv2.imread(self.path + "\\" + self.name))
