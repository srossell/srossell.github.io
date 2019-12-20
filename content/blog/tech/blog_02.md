Title: Blog 2
Date: 2018-05-7
Summary: ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu
Image: /images/blog/tech/blog_02/rf_lr.png
Tags: tag2
Slug: second-best

![title image](/images/blog/tech/blog_02/traces.png)

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.


En equation
$$I = f(x, \beta) + \Gamma $$

$$R(z) = max(0, z) =
     \begin{cases}
       0 &\quad\text{for } z\leq0 \\
       z &\quad\text{for } z > 0
     \end{cases}$$


$$\frac{d}{dz}\mathcal{R}(z)=
     \begin{cases}
       0 &\quad\text{for } z\leq0 \\
       1 &\quad\text{for } z > 0
     \end{cases}$$

$$W=
\begin{bmatrix}
    3 & -4 \\
    -2& 2\\
    0& 4
\end{bmatrix}
$$


$$I_0=
\begin{bmatrix}
    2 & 5 & 1
\end{bmatrix}
$$

with bias terms

$$b=
\begin{bmatrix}
    2 & -1
\end{bmatrix}
$$

Using the standard central-threshold neuron model, the output signal of the second layer is:

$$\mathcal{R}\Big(I_0\cdot W + b\Big) = \mathcal{R}\Big(
\begin{bmatrix}
    2 & 5 & 1
\end{bmatrix}
\cdot
\begin{bmatrix}
    3 & -4 \\
    -2& 2\\
    0& 4
\end{bmatrix}
+
\begin{bmatrix}
    2 & -1
\end{bmatrix}
\Big)
=
$$
$$
\mathcal{R}\Big(
\begin{bmatrix}
    -2 & 5
\end{bmatrix}
\Big)
=
\begin{bmatrix}
    \mathcal{R}(-2)& \mathcal{R}(5)
\end{bmatrix}
\Big)
=
\begin{bmatrix}
    0 & 5
\end{bmatrix}
$$

In the case of the multi-threshold neuron model proposed the output is

$$
[\sum_{i=1}^N\mathcal{R}(W_{i1}\cdot I_i) + b_1, \sum_{i=1}^N\mathcal{R}(W_{i2}\cdot I_i) + b_2]=
$$
$$
\begin{bmatrix}
    \mathcal{R}(6) + \mathcal{R}(-10) + \mathcal{R}(0) + 2 & \mathcal{R}(-8) + \mathcal{R}(10) + \mathcal{R}(4)  -1
\end{bmatrix}
=
\begin{bmatrix}
    8 & 13
\end{bmatrix}
$$



