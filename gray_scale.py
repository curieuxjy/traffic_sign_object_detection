import os
import glob
# from PIL import Image
import cv2

def exist_file():
    if (filename.lower().endswith('.png') or filename.lower().endswith('.jpg') or filename.lower().endswith('.gif') or filename.lower().endswith('.bmp') or filename.lower().endswith('.pcx')):
        return True

for filename in os.listdir('.'):
    if exist_file():    
        convert_gray()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        help="Directory path to raw images.",
        default="./data/raw",
        type=str,
    )
    parser.add_argument(
        "--save-dir",
        help="Directory path to save resized images.",
        default="./data/images",
        type=str,
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir
    save_dir = args.save_dir
    # assert isinstance(target_size, tuple) and len(target_size) == 2, msg
    fnames = glob.glob(os.path.join(raw_dir, "*.{}".format("png")))
    os.makedirs(save_dir, exist_ok=True)

    for i, fname in enumerate(fnames):
        print(".", end="", flush=True)

        # original_image = Image.open(fname)
        # original_image = original_image.convert('L')
        # new_fname = "{}.{}".format(str(i), '.jpg')
        # original_image.save(os.path.join(save_dir, new_fname))
        # original_image.close()  

        img = cv2.imread(fname)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_fname = "{}.{}".format(str(i), "jpg")
        gray_fname = os.path.join(save_dir, new_fname)
        cv2.imwrite(gray_fname, img_gray)



    print(
        "\nDone resizing {} files.\nSaved to directory: `{}`".format(
            len(fnames), save_dir
        )
    )
