```markdown
# DTW-Based Motion Recognition

## System Overview

1. Processes input video to extract pose landmarks (33 points)
2. Uses FastDTW with Euclidean distance for temporal alignment
3. Applies technique-specific thresholds for detection

## Key Features

- Automatic upside-down detection & correction
- Overlapping window segmentation (step=35 frames)
- Multi-class detection (14 techniques)

## Usage Notes

1. After launching the program, you will be prompted to enter the path to a video file. Ensure this path is an absolute path (relative paths may fail)

2. Annotations show:
   - Technique name
   - DTW distance
   - Detection timeframe

For an example video you can take videos from DTW/test_videos

## Expected Output

- Console: Recognized segments with frame ranges
- Video: Red bounding box + technique name during detection
```
