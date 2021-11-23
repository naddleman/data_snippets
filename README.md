# snippets

Little Python code snippets to demo various data analysis techniques

## Kernel smoothing

Implementing k-nearest neighbors (KNN) interpolation estimates a function.

Observe the estimated functions are jagged, with discontinuities whenever
a new point becomes one of the k used in calculating the average.

The discontinuities can be improved by kernel smoothing. In this case a
weighted average is taken of all points in a window of radius lambda.

The weights are given by a kernel function that decreases to zero at the
edges of the window, and so the estimated value at a point x_0 is

f(x_0) = (sum_{i} K(x_0, x_i) * y_i) / sum K(x_0, x_i)

In words, the prediction at x_0 is a weighted average of nearby observations.

A drawback of these local smoothing techniques is poor predictions at the
boundaries. Observe that all of these methods overestimate the function on
the left edge. In this region the function is increasing, and because it is
a boundary, the points that contribute to the average are all taken from
one side.


