import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from decord import VideoReader, cpu


def plot_images_grid(frames, video_path):
    """
    Plots a sequence of composite frames side-by-side in a single row 
    and saves the resulting figure as a PNG file.

    Args:
        - frames (np.ndarray): array of stacked/composite frames of shape (N, H, W, Channels).
        - video_path (str | list): path to the original video file. The base name 
            of this path is used to name the output PNG file.
            
    Returns:
        None
    """
    # Unpack the dimensions of the frames array
    num_frames, img_H, img_W, _ = frames.shape

    # Calculate figure size to maintain the original aspect ratio of the frames
    aspect_ratio = img_W / img_H
    # Scale width by aspect ratio and an arbitrary multiplier (5) for readability
    fig_width = num_frames * aspect_ratio * 5
    # Set height consistently with the width multiplier
    fig_height = 5

    # Initialize the matplotlib figure and subplots
    fig, axes = plt.subplots(1, num_frames, figsize=(fig_width, fig_height))
    
    # Ensure axes is always a 2D array, even if h=1 and w=1, to prevent indexing errors
    axes = np.atleast_2d(axes)

    # Loop through the grid to plot frames
    for idx in range(num_frames):
        axes[0, idx].imshow(frames[idx])
        axes[0, idx].axis('off')

    # Adjust the spacing between frames (wspace=horizontal, hspace=vertical)
    plt.subplots_adjust(wspace=0.1, hspace=0.01)
    
    # Extract the filename from the path and remove the extension (e.g., .mp4)
    if isinstance(video_path, list):
        video_path = video_path[0]  # Handle list case just in case
    video_name = os.path.basename(video_path)
    video_name = os.path.splitext(video_name)[0]
    
    # Save the figure as a PNG with tight bounding boxes to remove excess white space
    plt.savefig(video_name + '.png', pad_inches=0.1, bbox_inches='tight')

def stack_frames_grid(video, panel_width, panel_height, border_px):
    """
    Reshapes a video by stacking (panel_width x panel_height) frames into a grid 
    with optional black borders both between and around panels. The final resolution 
    remains the same as the input frames.

    Args:
        - video (np.ndarray): input video array of shape (N, H, W, C)
        - panel_width (int): number of frames per row.
            alpha in the paper.
        - panel_height (int): number of frames per column.
            beta in the paper.
        - border_px (int): border size in pixels (around frames).
            Suggested value: 0
        - fps_limit (int): how many fps between frames to sample

    Returns:
        - stacked_video (np.ndarray): reshaped video of shape (N // (w*h), H, W, Channels).
        
    Raises:
        AssertionError: If the total number of frames (N) is not perfectly divisible 
            by the number of panels per grid (panel_width * panel_height).
    """
    # Load paneling dimensions
    w, h = panel_width, panel_height
    border_px = border_px

    # Get original video dimensions
    D, H, W, Channels = video.shape
    frames_per_grid = w * h
    assert D % frames_per_grid == 0, f"Number of frames ({D}) must be divisible by w*h ({w*h})"

    ### Compute size for each panel image ###
    
    # Calculate the total pixels consumed by borders (panels + 1 for outer edges)
    total_border_w = (w + 1) * border_px
    total_border_h = (h + 1) * border_px

    # Calculate the size of the individual scaled-down frames
    panel_W = (W - total_border_w) // w
    panel_H = (H - total_border_h) // h

    # Calculate total frames of the new output video
    new_num_frames = D // frames_per_grid
    
    # Initialize output array with black background (zeros). 
    # The empty spaces between panels will naturally remain black.
    stacked_video = np.zeros((new_num_frames, H, W, Channels), dtype=video.dtype) 

    for i in range(new_num_frames):
        # Extract the chunk of frames that will be combined into a single grid frame
        grid_frames = video[i * frames_per_grid: (i + 1) * frames_per_grid]
        
        # Create a black canvas for the current composite frame
        frame_canvas = np.zeros((H, W, Channels), dtype=video.dtype) 

        # Iterate over all the chunk of frames to create the grid
        for idx in range(frames_per_grid):
            # Determine the grid position (row and column) for the current frame
            row = idx // w
            col = idx % w

            # Resize the individual frame to fit the calculated panel dimensions
            resized = cv2.resize(grid_frames[idx], (panel_W, panel_H), interpolation=cv2.INTER_AREA)

            # Calculate placement coordinates
            y = (row + 1) * border_px + row * panel_H
            x = (col + 1) * border_px + col * panel_W

            # Paste the resized frame onto the canvas
            frame_canvas[y:y+panel_H, x:x+panel_W] = resized

        # Store the finished composite frame in the output array
        stacked_video[i] = frame_canvas

    return stacked_video


def load_video(video_path, C, panel_width=2, panel_height=2, border_px=0, fps_limit=1, verbose=False, plot_video=False):
    '''
    Load video frames using either a paneled (grid) approach or normal uniform sampling.

    Args:
        - video_path (str | list): path to the video(s) file. 
            If a list is provided, only the first video will be loaded.
        - C (int): maximum number of final frames to load (VLM context window).
            If paneling occurs, the method internally extracts C * (width * height) 
            frames before stacking them into C paneled frames.
        - panel_width (int): number of frames per row.
            alpha in the paper.
        - panel_height (int): number of frames per column.
            beta in the paper.
        - border_px (int): border size in pixels (around frames).  
            Suggested value: 0
        - fps_limit (int): how many fps between frames to sample
        - verbose (bool): whether to print verbose output
        - plot_video (bool): whether to plot the video frames

    Returns:
        - spare_frames (np.ndarray): the loaded video frames. 
            Shape is (C, H, W, Channels), where C is the requested number of frames.
            
    Raises:
        - ValueError: if video_path is not a string or a list of strings.
    '''
    # Load the video with decord
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    elif type(video_path) == list:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    else:
        raise ValueError("video_path should be a string or a list of strings")

    # Get total frame number of the video (D in the paper)
    D = len(vr)

    # Store original max frames number
    C_original = C
    # Compute the total frames needed for paneling
    C = C * panel_width * panel_height

    # Compute the offset for frame sampling (gamma in the paper)
    offset = fps_limit * vr.get_avg_fps()

    # Check if there are enough frames, spaced at least "offset" frames apart
    if (D > offset * C) and (panel_width * panel_height > 1):
        '''PANELING'''
        # Uniformly sample C frames, from start to end of the video
        # First compute the frame indices, then sample the frames
        uniform_sampled_frames = np.linspace(0, D - 1, C, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()

        # Panel the sampled frames
        spare_frames = stack_frames_grid(spare_frames, panel_width, panel_height, border_px)

        # For debugging purposes
        if verbose:
            print('Paneled:', spare_frames.shape, panel_width, panel_height, D, vr.get_avg_fps(), D / vr.get_avg_fps(), C)
    else:
        '''NORMAL SAMPLING'''
        # Uniformly sample C frames, from start to end of the video
        # First compute the frame indices, then sample the frames
        uniform_sampled_frames = np.linspace(0, D - 1, C_original, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()

        # For debugging purposes
        if verbose:
            print('Sampled:', spare_frames.shape, D)

    # Plot the video frames
    if plot_video:
        plot_images_grid(spare_frames, video_path)
    
    return spare_frames