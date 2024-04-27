#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#include "matrix.h"
#include <time.h>

// Frees an array of descriptors.
// descriptor *d: the array.
// int n: number of elements in array.
void free_descriptors(descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(d[i].data);
    }
    free(d);
}

// Create a feature descriptor for an index in an image.
// image im: source image.
// int i: index in image for the pixel we want to describe.
// returns: descriptor for that index.
descriptor describe_index(image im, int i)
{
    int w = 5;
    descriptor d;
    d.p.x = i%im.w;
    d.p.y = i/im.w;
    d.data = calloc(w*w*im.c, sizeof(float));
    d.n = w*w*im.c;
    int c, dx, dy;
    int count = 0;
    // If you want you can experiment with other descriptors
    // This subtracts the central value from neighbors
    // to compensate some for exposure/lighting changes.
    for(c = 0; c < im.c; ++c){
        float cval = im.data[c*im.w*im.h + i];
        for(dx = -w/2; dx < (w+1)/2; ++dx){
            for(dy = -w/2; dy < (w+1)/2; ++dy){
                float val = get_pixel(im, i%im.w+dx, i/im.w+dy, c);
                d.data[count++] = cval - val;
            }
        }
    }
    return d;
}

// Marks the spot of a point in an image.
// image im: image to mark.
// ponit p: spot to mark in the image.
void mark_spot(image im, point p)
{
    int x = p.x;
    int y = p.y;
    int i;
    for(i = -9; i < 10; ++i){
        set_pixel(im, x+i, y, 0, 1);
        set_pixel(im, x, y+i, 0, 1);
        set_pixel(im, x+i, y, 1, 0);
        set_pixel(im, x, y+i, 1, 0);
        set_pixel(im, x+i, y, 2, 1);
        set_pixel(im, x, y+i, 2, 1);
    }
}

// Marks corners denoted by an array of descriptors.
// image im: image to mark.
// descriptor *d: corners in the image.
// int n: number of descriptors to mark.
void mark_corners(image im, descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        mark_spot(im, d[i].p);
    }
}

// Creates a 1d Gaussian filter.
// float sigma: standard deviation of Gaussian.
// returns: single row image of the filter.
image make_1d_gaussian(float sigma)
{
    // TODO: make separable 1d Gaussian.
    int size = ceil(6 * sigma);
    if (size % 2 == 0) size++;
    int center = size / 2;

    image filter = make_image(size, 1, 1);

    float sum = 0;
    for (int x = 0; x < size; ++x) {
        float dx = x - center;
        float distance = dx * dx;

        float weight = exp(-distance / (2 * sigma * sigma)) / (TWOPI * sigma * sigma);
        
        set_pixel(filter, x, 0, 0, weight);
        sum += weight;
    }

    for (int i = 0; i < size; ++i) {
        filter.data[i] /= sum;
    }
    return filter;
}

image make_1d_gaussian_v(float sigma)
{
    // TODO: make separable vertical 1d Gaussian.
    int size = ceil(6 * sigma);
    if (size % 2 == 0) size++;
    int center = size / 2;

    image filter = make_image(1, size, 1);

    float sum = 0;
    for (int y = 0; y < size; ++y) {
        float dy = y - center;
        float distance = dy * dy;

        float weight = exp(-distance / (2 * sigma * sigma)) / (TWOPI * sigma * sigma);
        
        set_pixel(filter, 0, y, 0, weight);
        sum += weight;
    }

    for (int i = 0; i < size; ++i) {
        filter.data[i] /= sum;
    }
    return filter;
}

// Smooths an image using separable Gaussian filter.
// image im: image to smooth.
// float sigma: std dev. for Gaussian.
// returns: smoothed image.
image smooth_image(image im, float sigma)
{
    // TODO: use two convolutions with 1d gaussian filter.
    image filter = make_1d_gaussian(sigma);
    image filter_v = make_1d_gaussian_v(sigma);
    image horizontal_blur = convolve_image(im, filter, 0);
    image smooth = convolve_image(horizontal_blur, filter_v, 0);

    free_image(filter);
    free_image(filter_v);
    free_image(horizontal_blur);

    return smooth;
}

// Multiply corresponding pixels of two images element-wise
image mult_image(image a, image b)
{
    // Make sure both images have the same dimensions
    assert(a.w == b.w && a.h == b.h && a.c == b.c);

    // Create a new image to store the result
    image result = make_image(a.w, a.h, a.c);

    // Perform element-wise multiplication
    for (int i = 0; i < a.w * a.h * a.c; ++i) {
        result.data[i] = a.data[i] * b.data[i];
    }

    return result;
}


// Calculate the structure matrix of an image.
// image im: the input image.
// float sigma: std dev. to use for weighted sum.
// returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
//          third channel is IxIy.
image structure_matrix(image im, float sigma)
{
    // TODO: calculate structure matrix for im.
    image gx = make_gx_filter();
    image gy = make_gy_filter();
    image Ix = convolve_image(im, gx, 0);
    image Iy = convolve_image(im, gy, 0);
    free_image(gx);
    free_image(gy);

    // Calculate measures
    image Ix2 = mult_image(Ix, Ix);   // Ix^2
    image Iy2 = mult_image(Iy, Iy);   // Iy^2
    image Ixy = mult_image(Ix, Iy);   // Ix * Iy

    // Smooth the measures with a Gaussian filter
    Ix2 = smooth_image(Ix2, sigma);
    Iy2 = smooth_image(Iy2, sigma);
    Ixy = smooth_image(Ixy, sigma);

    // Create the structure matrix S
    image S = make_image(im.w, im.h, 3);
    for (int i = 0; i < im.w * im.h; ++i) {
        S.data[i] = Ix2.data[i];               // First channel: Ix^2
        S.data[i + im.w * im.h] = Iy2.data[i]; // Second channel: Iy^2
        S.data[i + 2 * im.w * im.h] = Ixy.data[i]; // Third channel: Ix * Iy
    }

    // Free the memory allocated for the intermediate images
    free_image(Ix);
    free_image(Iy);
    free_image(Ix2);
    free_image(Iy2);
    free_image(Ixy);

    return S;

}

// Estimate the cornerness of each pixel given a structure matrix S.
// image S: structure matrix for an image.
// returns: a response map of cornerness calculations.
image cornerness_response(image S)
{
    image R = make_image(S.w, S.h, 1);
    // TODO: fill in R, "cornerness" for each pixel using the structure matrix.
    // We'll use formulation det(S) - alpha * trace(S)^2, alpha = .06.
    float alpha = 0.06;
    for (int i = 0; i < S.w * S.h; ++i) {
        float ix2 = S.data[i];
        float iy2 = S.data[i + S.w * S.h];
        float ixiy = S.data[i + 2 * S.w * S.h];
        float det = ix2 * iy2 - ixiy * ixiy;
        float trace = ix2 + iy2;
        float R_val = det - alpha * trace * trace;
        R.data[i] = R_val;
    }
    
    return R;
}

// Perform non-max supression on an image of feature responses.
// image im: 1-channel image of feature responses.
// int w: distance to look for larger responses.
// returns: image with only local-maxima responses within w pixels.
image nms_image(image im, int w)
{
    image r = copy_image(im);
    // TODO: perform NMS on the response map.
    // for every pixel in the image:
    //     for neighbors within w:
    //         if neighbor response greater than pixel response:
    //             set response to be very low (I use -999999 [why not 0??])
    for (int i = 0; i < im.w * im.h; ++i) {
        int row = i / im.w;
        int col = i % im.w;
        float val = get_pixel(im, col, row, 0);
        for (int x = col - w; x <= col + w; ++x) {
            for (int y = row - w; y <= row + w; ++y) {
                if (x >= 0 && x < im.w && y >= 0 && y < im.h) {
                    if (get_pixel(im, x, y, 0) > val) {
                        set_pixel(r, col, row, 0, -999999);
                    }
                }
            }
        }
    }
    
    return r;
}

// Perform harris corner detection and extract features from the corners.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
// int *n: pointer to number of corners detected, should fill in.
// returns: array of descriptors of the corners in the image.
descriptor *harris_corner_detector(image im, float sigma, float thresh, int nms, int *n)
{
    // Calculate structure matrix
    image S = structure_matrix(im, sigma);

    // Estimate cornerness
    image R = cornerness_response(S);

    // Run NMS on the responses
    image Rnms = nms_image(R, nms);


    //TODO: count number of responses over threshold
    int count = 0; // change this
    for (int i = 0; i < im.w * im.h; ++i) {
        if (Rnms.data[i] > thresh) count++;
    }
    
    *n = count; // <- set *n equal to number of corners in image.
    descriptor *d = calloc(count, sizeof(descriptor));

    if (count == 0) return d;

    int idx = 0;
    //TODO: fill in array *d with descriptors of corners, use describe_index.
    for (int i = 0; i < Rnms.w; ++i) {
        for (int j = 0; j < Rnms.h; ++j) {
            if (get_pixel(Rnms, i, j, 0) > thresh) {
                int offset = j * Rnms.w + i;
                d[idx++] = describe_index(im, offset);
            }
        }
    }

    free_image(S);
    free_image(R);
    free_image(Rnms);
    return d;
}

// Find and draw corners on an image.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
void detect_and_draw_corners(image im, float sigma, float thresh, int nms)
{
    int n = 0;
    descriptor *d = harris_corner_detector(im, sigma, thresh, nms, &n);
    mark_corners(im, d, n);
}
