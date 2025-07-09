# Task-5 Image Compression using Matrix Factorisation

You have an image patch of size (50x50) that you want to compress using matrix factorization. To do this, you'll split the patch [N×N] into two smaller matrices of size [N×r] and [r×N] using matrix factorization. Compute the compressed patch by multiplying these two matrices and compare the reconstructed image patch with the original patch. Compute the Root Mean Squared Error (RMSE) and Peak Signal-to-Noise Ratio (PSNR) between the original and reconstructed image patches.

- Test different values for the low-rank r=[5,10,25,50].
- Use Gradient Descent to learn the compressed matrices.
- Display the reconstructed image patches, keeping the original pixel values outside the patch unchanged, and use your compressed matrix for the patch to show how well the reconstruction works.
- Compute the RMSE and PSNR for each value of r.


## Solution:

Matrix factorization is a technique used to compress image data by approximating the image matrix with lower-dimensional components, while retaining the overall structure of the image. Instead of reducing resolution, it compresses the necessary information to represent the image effectively.

## How Image Compression Works?

1. **Image Matrix Representation**: An image is represented as a matrix `A` of size $(M \times N)$, where $M$ is the number of rows (height) and $N$ is the number of columns (width).

2. **Factorization**: The original matrix `A` is factorized into two smaller matrices:
   - `W` of size $(M \times r)$
   - `H` of size $(r \times N)$
   
   Here, $r$ is a rank that is significantly smaller than $M$ or $N$. This allows for effective compression while maintaining an approximation of the original image.


Without compression, the original image matrix requires storing **M \* N** values, representing every pixel. However, by applying matrix factorization, the storage requirement drops to **(M \* r) + (r \* N)** values, significantly reducing the space needed. As the rank **r** decreases, the compression ratio increases, allowing for more efficient storage. However, this comes at the cost of image quality, as lower ranks lead to a loss of fine details, which can be noticed as blurring.

## Image Compression
First, we extracted three 50x50 patches from the image and then performed image compression and matrix factorization on each one of them with the varying rank = [5,10,25,50].

## Patch-1

![alt text](image.png)
- **Rank = 5**

![alt text](image-1.png) 
![alt text](image-2.png)
    ```
    RMSE: 0.006451
    PSNR: 43.807847
    ```

- **Rank = 10**

![alt text](image-3.png)   
![alt text](image-4.png)
    ```
    RMSE: 0.006834
    PSNR: 43.307004
    ```

- **Rank = 20**

![alt text](image-5.png)    
![alt text](image-6.png)
    ```
    RMSE: 0.003929
    PSNR: 48.114580
    ```

- **Rank = 50**

![alt text](image-7.png) 
![alt text](image-8.png)
    ```
    RMSE: 0.003483
    PSNR: 49.161294
    ```

Graphs illustrating RMSE vs Rank and PSNR vs Rank for Patch-1 are displayed below:

![alt text](image-9.png)

## Patch-2

![alt text](image-10.png)

- **Rank = 5**
![alt text](image-11.png)
![alt text](image-12.png)
    ```
    RMSE: 0.030466
    PSNR: 30.323810
    ```

- **Rank = 10**
![alt text](image-13.png)
![alt text](image-14.png)
    ```
    RMSE: 0.025922
    PSNR: 31.726516
    ```

- **Rank = 20**

![alt text](image-15.png)   
![alt text](image-16.png)
    ```
    RMSE: 0.015575
    PSNR: 36.151359
    ```

- **Rank = 50**

![alt text](image-17.png) 
![alt text](image-18.png)
    ```
    RMSE: 0.009523
    PSNR: 40.424417
    ```

Graphs illustrating RMSE vs Rank and PSNR vs Rank for Patch-2 are displayed below:

![alt text](image-19.png)

## Patch-3

![alt text](image-20.png)

- **Rank = 5**

![alt text](image-21.png)  
![alt text](image-22.png)
    ```
    RMSE: 0.031112
    PSNR: 30.141574
    ```

- **Rank = 10**

![alt text](image-23.png) 
![alt text](image-24.png)
    ```
    RMSE: 0.023612
    PSNR: 32.537242
    ```

- **Rank = 20**

![alt text](image-25.png)
![alt text](image-26.png)
    ```
    RMSE: 0.017403
    PSNR: 35.187531
    ```

- **Rank = 30**

![alt text](image-27.png)
![alt text](image-28.png)
    ```
    RMSE: 0.008954
    PSNR: 40.960107
    ```
- **Rank = 50**

![alt text](image-29.png)

![alt text](image-30.png)
Graphs illustrating RMSE vs Rank and PSNR vs Rank for Patch-3 are displayed below:

![alt text](image-31.png)
