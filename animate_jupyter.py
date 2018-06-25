import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc
from IPython.display import HTML
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\woottenm\FFMPEG\bin\ffmpeg.exe'

def animate_image_sequence(images, interval=20, show_frame=False):
    fig, ax = plt.subplots()
    img = plt.imshow(images[0])

    def init():
        img.set_data(images[0])
        return (img,)
    def animate(i):
        img.set_data(images[i])
        if show_frame:
            plt.title('Frame {:04d}'.format(i))
        return (img,)
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(images),
        interval=interval,
        blit=True
    )
    return HTML(anim.to_html5_video())

def animate_image_sequence_pair(images_1, images_2, interval=20, show_frame=False):
    fig, [ax1, ax2] = plt.subplots(1, 2)
    ax1.axis('off')
    ax2.axis('off')

    img1 = ax1.imshow(images_1[0])
    img2 = ax2.imshow(images_2[0])


    def init():
        img1.set_data(images_1[0])
        img2.set_data(images_2[0])
        return (img1, img2)
    def animate(i):
        img1.set_data(images_1[i])
        img2.set_data(images_2[i])
        if show_frame:
            plt.title('Frame {:04d}'.format(i))
        return (img1, img2)

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(images_1),
        interval=interval,
        blit=True
    )
    return HTML(anim.to_html5_video())
