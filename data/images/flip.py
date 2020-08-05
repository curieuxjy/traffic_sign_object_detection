import os
import glob
import cv2

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Resize raw images to uniformed target size."
    )
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
    
    fnames = glob.glob(os.path.join(raw_dir, "*.{}".format("jpg")))
    os.makedirs(save_dir, exist_ok=True)
    
    for i, fname in enumerate(fnames):
        print(".", end="", flush=True)
        img = cv2.imread(fname)
        img_flip = cv2.flip(img, 1)
        new_fname = "{}".format(fname.split("\\")[1])
        flip_fname = os.path.join(save_dir, new_fname)
        cv2.imwrite(flip_fname, img_flip)
    print(
        "\nDone Flip {} files.\nSaved to directory: `{}`".format(
            len(fnames), save_dir
        )
    )
