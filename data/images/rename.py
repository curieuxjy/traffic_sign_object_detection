import os
import glob
import cv2

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rename raw images to uniformed target size."
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
    # parser.add_argument(
    #     "--ext", help="Raw image files extension to resize.", default="jpg", type=str
    # )
    parser.add_argument(
        "--target-size",
        help="Target size to resize as a tuple of 2 integers.",
        default="(720, 720)",
        type=str,
    )

    parser.add_argument(
        "--class-name",
        help="Target class",
        default="none",
        type=str,
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir
    save_dir = args.save_dir
    target_size = eval(args.target_size)
    class_name = args.class_name

    msg = "--target-size must be a tuple of 2 integers"
    assert isinstance(target_size, tuple) and len(target_size) == 2, msg

    fnames = glob.glob(os.path.join(raw_dir, "*.{}".format("jpg")))
    os.makedirs(save_dir, exist_ok=True)
    print(
        "{} files to resize from directory `{}` to target size:{}".format(
            len(fnames), raw_dir, target_size
        )
    )

    plus_num = 0
    if class_name = "bicycle":
        plus_num = 0
    elif class_name = "bump":
        plus_num = 0
    elif class_name = "child":
        plus_num = 0
    elif class_name = "const":
        plus_num = 0
    elif class_name = "cross":
        plus_num = 0




    for i, fname in enumerate(fnames):
        print(".", end="", flush=True)
        img = cv2.imread(fname)
        img_small = cv2.resize(img, target_size)
        new_fname = "{}.{}".format(str(i+plus_num), "jpg")
        small_fname = os.path.join(save_dir, new_fname)
        cv2.imwrite(small_fname, img_small)
    print(
        "\nDone resizing {} files.\nSaved to directory: `{}`".format(
            len(fnames), save_dir
        )
    )
