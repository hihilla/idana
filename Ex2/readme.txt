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
   3) Sobel-like operator that results with the edges "lifted" (giving it 3D
   feeling) from the image.
   the kernel metrix:
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
c. Since the kernel is 3X3 matrix the need for padding is only for one layer
"around" the images itself; 1 row from above and one row from beneath the image,
and one column from each side of the image. Because we wanted to have
"soft padding", meaning no drastic changes while padding, the added vectors
around the image were reflections of the image itself.