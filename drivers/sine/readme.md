Data for approximating the sine function y : [-pi,pi] -> [-1,1], y(x) = sin(x)

Data contain 1000 randomly sampled x-values in [-pi,pi] (uniformly sampled, "x.dat"), and corresponding y-values with y = sin(x) ("y.dat")

Network:
* Network width of d=2
* Opening layer copies 1d x-data to network width, i.e. W_in = [1;1], b_in = [0;0], no activation.
* Closing layer averages over the network's width d:
  g(y) = 1/d sum_d y_d
  i.e. W_out = [0.5 0.5], b_out = [0 0], no activation
* Intermediate layers are dense 2x2 weights matrices and a 2x1 bias vector, Activation is tanh
* Loss function computes L2-error:
    loss(y_data,g(y)) = 1/2 || y_data - g(y) ||^2 
* For Resnet, 10-15 layers should be enough to get the loss down to at least 1e-4