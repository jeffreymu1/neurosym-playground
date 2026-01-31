playing around with 3d models

backbone: 3d assets (partnet) -> rgb renders from diff perspectives w/ intrinsics+extrinsics -> run sam vit to segment rgb images -> use extrinsic depth info to backproject 2d segments into 3d -> fuse consistent clusters of objects into one -> good part segmentations to feed into neurosym model

