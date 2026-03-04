"""
Constants for self-supervised learning (SSL) task positions.

These define the index positions for different SSL pretext tasks
in the label array used during multi-task learning.
"""

# SSL Task Positions in Label Array
TIME_REVERSAL_POS = 0    # Time reversal (temporal flip) task
SCALE_POS = 1            # Scaling transformation task
PERMUTATION_POS = 2      # Segment permutation task
TIME_WARPED_POS = 3      # Time warping task
