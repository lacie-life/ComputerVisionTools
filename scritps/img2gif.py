import glob
import contextlib
from PIL import Image

data_path = '/home/lacie/Datasets/A9_dataset/TrafCam_format/pre_trained_outputs/'

def make_gif(frame_folder):
    paths = glob.glob(f"{frame_folder}/*.jpg")
    paths.sort()
    frames = [Image.open(image).resize(((480, 480))) for image in paths]
    frame_one = frames[0]
    frame_one.save("pre_train.gif", format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)


if __name__ == "__main__":
    make_gif(data_path)