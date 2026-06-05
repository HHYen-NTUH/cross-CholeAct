import os

from kinetics import VideoClsDataset


def build_dataset(is_train, test_mode, args):
    """Build the cross-CholeAct clip-classification split.

    ``args.data_path`` must contain VideoMAE-style CSV files:

      train.csv
      val.csv
      test.csv

    Each row is space-delimited: ``/absolute/or/relative/clip.mp4 label``.
    """
    if is_train:
        mode = "train"
        anno_path = os.path.join(args.data_path, "train.csv")
    elif test_mode:
        mode = "test"
        anno_path = os.path.join(args.data_path, "test.csv")
    else:
        mode = "validation"
        anno_path = os.path.join(args.data_path, "val.csv")

    dataset = VideoClsDataset(
        anno_path=anno_path,
        data_path="/",
        mode=mode,
        clip_len=args.num_frames,
        frame_sample_rate=args.sampling_rate,
        num_segment=1,
        test_num_segment=args.test_num_segment,
        test_num_crop=args.test_num_crop,
        num_crop=1 if not test_mode else 3,
        keep_aspect_ratio=True,
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        new_height=256,
        new_width=320,
        args=args,
    )
    print("Number of the class = %d" % args.nb_classes)
    return dataset, args.nb_classes
