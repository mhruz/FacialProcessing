import dlib
import glob
import os
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect facial landmarks in dir of images.')
    parser.add_argument('path_images', type=str, help='path to images to detect facial keypoints in')
    parser.add_argument('path_to', type=str, help='full-path where to save the data (json)')
    parser.add_argument('--data_range', type=str, help='data range in format - from:to (eg. 0:100)', default="0:")
    parser.add_argument('--verbose', type=int,
                        help='level of verbosity: 0 - nothing; 1 - draw detections and wait for \'enter\'', default=0)
    args = parser.parse_args()

    predictor_path = "models/shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    if args.verbose == 1:
        win = dlib.image_window()

    extentions = ("jpg", "jp2", "png", "bmp", "jpeg", "tiff")

    image_files = os.listdir(args.path_images)
    image_files = [x for x in image_files if x.lower().endswith(extentions)]

    data = {}
    data_range = args.data_range.split(":")
    if data_range[0] == "":
        data_range[0] = "0"

    if data_range[1] == "":
        data_range[1] = "{}".format(len(image_files))

    data_range = [int(x) for x in data_range]

    for idx, f in enumerate(image_files[data_range[0]:data_range[1]]):
        print("Processing file {}: {}".format(idx, f))
        img = dlib.load_rgb_image(os.path.join(args.path_images, f))

        if args.verbose == 1:
            win.clear_overlay()
            win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))

        filename = os.path.basename(f)

        if len(dets) > 0:
            data[filename] = []

        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                      shape.part(1)))
            # Draw the face landmarks on the screen.
            if args.verbose == 1:
                win.add_overlay(shape)

            shape_dict = {}
            for i in range(shape.num_parts):
                shape_dict[i] = {"x": shape.part(i).x / img.shape[1], "y": shape.part(i).y / img.shape[0]}

            data[filename].append({"bbox": {"left": d.left() / img.shape[1], "top": d.top() / img.shape[0],
                                            "right": d.right() / img.shape[1], "bottom": d.bottom() / img.shape[0]},
                                   "shape": shape_dict})

        if args.verbose == 1:
            win.add_overlay(dets)
            dlib.hit_enter_to_continue()

    with open(args.path_to, "wt") as f_out:
        f_out.write(json.dumps(data, indent=3))
