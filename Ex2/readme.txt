Adar Lavi 308491596
Hilla Barkal 200226132

Task 2
a. 1) backward difference: edge enhance. the kernel matrix:
[[0, 0, 0],
 [-1,1, 0],
 [0, 0, 0]]
   2) forward difference: a different edge enhance. the kernel matrix:
[[0, 0, 0],
 [0,-1, 1],
 [0, 0, 0]]
   3) Sobel-like operator that results with the edges "lifted" from the image. the kernel metrix:
[[-1, 0, 1],
 [-2, 0, 2],
 [ 1, 0, 1]]
b. decompose Sobel operator:
 x-direction:
[[1, 0, -1],
 [2, 0, -2],
 [1, 0, -1]]
this can be separated to [1, 2, 1] * [1, 0, -1].T
 y-direction:
[[1, 2, 1],
 [0, 0, 0],
 [-1,-2,-1]]
this can be separated to [1, 0, -1] * [1, 2, 1].T
therefore the directional gradient of an image A can be computed:
Gx = [1, 2, 1].T * ([1, 0, -1] * A)
Gy = [1, 0, -1].T * ([1, 2, 1] * A)
c. approach