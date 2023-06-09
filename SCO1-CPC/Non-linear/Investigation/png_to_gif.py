import glob
import contextlib
from PIL import Image

# filepaths
fp_in = glob.glob("plots/heatmaps/*.png")
fp_out = "plots/heatmap.gif"

# use exit stack to automatically close opened images
with contextlib.ExitStack() as stack:
    # lazily load images
    imgs = (stack.enter_context(Image.open(f)) for f in fp_in)
    # extract  first image from iterator
    img = next(imgs)
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=400, loop=0)