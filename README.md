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

## Local Linear Regression

Local linear regression circumvents this sort of bias by fitting a line with
least squares to the kernel-weighted data points for every prediction.

The implementation in this script sacrifices efficiency for expressiveness.

Instead of minimizing the residual sum of squares as usual, for each predicted
point (x0) we find the linear fit, α, β in f(x0) = α + β * x0 that minimizes
the residuals weighted by distance from the point by some kernel function.

Notice that the bias on the left side has disappeared. Better predictions at
boundaries is a big advantage of the local linear fit over the local constant
of the weighted averages.
