import numpy as np
import cv2
# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    h, w = img.shape[0:2]
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = w
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = h

    # Compute the span of the region to be searched
    span_x = x_start_stop[1] - x_start_stop[0]
    span_y = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    num_pix_x = int(xy_window[0] * (1 - xy_overlap[0]))
    num_pix_y = int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    num_win_x = 1 + (span_x - xy_window[0]) // num_pix_x
    num_win_y = 1 + (span_y - xy_window[1]) // num_pix_y
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for i_y in range(num_win_y):
        for i_x in range(num_win_x):
            # Note: you could vectorize this step, but in practice
            # you'll be considering windows one by one with your
            # classifier, so looping makes sense
            # Calculate each window position
            upper_left = (x_start_stop[0] + i_x * num_pix_x, y_start_stop[0] + i_y * num_pix_y)
            lower_right = (xy_window[0] + upper_left[0], xy_window[1] + upper_left[1])
            window_list.append((upper_left, lower_right))
            # Append window position to list
    # Return the list of windows
    return window_list
