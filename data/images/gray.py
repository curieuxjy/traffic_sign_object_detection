import os
import glob
# from PIL import Image
import cv2

# def exist_file():
#     if (filename.lower().endswith('.png') or filename.lower().endswith('.jpg') or filename.lower().endswith('.gif') or filename.lower().endswith('.bmp') or filename.lower().endswith('.pcx')):
#         return True
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
    fnames = glob.glob(os.path.join(raw_dir, "*.{}".format("jpg")))
    os.makedirs(save_dir, exist_ok=True)

    for i, fname in enumerate(fnames):
        print(".", end="", flush=True)

        img = cv2.imread(fname)
        print(fname)
        print(type(fname))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_fname = "{}".format(fname.split("\\")[1])
        gray_fname = os.path.join(save_dir, new_fname)
        cv2.imwrite(gray_fname, img_gray)

    print(
        "\nMake GRAY scale {} files.\nSaved to directory: `{}`".format(
            len(fnames), save_dir
        )
    )
