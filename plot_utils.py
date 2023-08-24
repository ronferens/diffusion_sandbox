import tempfile

import matplotlib.pyplot as plt
import torch
from os.path import join
import cv2


def cvt_img_for_display(in_img):
    img = (in_img / 2 + 0.5).clamp(0, 1).squeeze()
    img = (img.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
    return img


def animate_sampling(intermediate_imgs, save_path='', prefix=None, fps=30):
    fig = None
    image_list = []

    with tempfile.TemporaryDirectory() as tmpdirname:
        for t in range(len(intermediate_imgs)):
            fig = plt.figure()
            plt.imshow(cvt_img_for_display(intermediate_imgs[t]))
            plt.title(f'Index={t}')
            tmp_filename = f'{t}.png'
            fig.savefig(join(tmpdirname, tmp_filename))
            image_list.append(tmp_filename)
            plt.close(fig)

        if len(image_list) == 0:
            raise ValueError("No images found in the specified folder.")

        # Get the first image to determine dimensions
        first_image = cv2.imread(join(tmpdirname, image_list[0]))
        height, width, _ = first_image.shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can also use other codecs like 'MJPG', 'H264', etc.
        video_name = join(save_path, 'denoising_progress.avi' if prefix is None else f'{prefix}_denoising_progress.avi')
        out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

        for image_file in image_list:
            image_path = join(tmpdirname, image_file)
            frame = cv2.imread(image_path)

            if frame is None:
                continue

            out.write(frame)

        # Release the VideoWriter and destroy any OpenCV windows
        out.release()
        cv2.destroyAllWindows()