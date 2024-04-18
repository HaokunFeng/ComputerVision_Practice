#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#define TWOPI 6.2831853
#define pi 3.14159265358979323846

/******************************** Resizing *****************************
  To resize we'll need some interpolation methods and a function to create
  a new image and fill it in with our interpolation methods.
************************************************************************/

float nn_interpolate(image im, float x, float y, int c)
{
    // TODO
    /***********************************************************************
      This function performs nearest-neighbor interpolation on image "im"
      given a floating column value "x", row value "y" and integer channel "c",
      and returns the interpolated value.
    ************************************************************************/
    // find the nearest neighbor pixel
    int nearest_x = (int)(x + 0.5);
    int nearest_y = (int)(y + 0.5);

    // ensure indices are within bounds
    nearest_x = (nearest_x < 0) ? 0 : ((nearest_x >= im.w) ? im.w - 1 : nearest_x);
    nearest_y = (nearest_y < 0) ? 0 : ((nearest_y >= im.h) ? im.h - 1 : nearest_y);

    // get the pixel value
    float interpolated_value = get_pixel(im, nearest_x, nearest_y, c);

    return interpolated_value;

}

image nn_resize(image im, int w, int h)
{
    // TODO Fill in (also fix the return line)
    /***********************************************************************
      This function uses nearest-neighbor interpolation on image "im" to a new
      image of size "w x h"
    ************************************************************************/
    // create a new image of size w x h
    image resized = make_image(w, h, im.c);

    // compute the scaling factor
    float sx = (float)(im.w) / w;
    float sy = (float)(im.h) / h;

    // loop over the new image and fill in the pixel values
    for (int y = 0; y < h; y++){
      for (int x = 0; x < w; x++){
        float old_x = (x + 0.5) * sx - 0.5;
        float old_y = (y + 0.5) * sy - 0.5;

        for (int c = 0; c < im.c; c++){
          float value = nn_interpolate(im, old_x, old_y, c);
          set_pixel(resized, x, y, c, value);
        }
      }
    }
    return resized;
}

/*
int clamp_int(int value, int min_value, int max_value)
{
    if (value < min_value) {
        return min_value;
    } else if (value > max_value) {
        return max_value;
    } else {
        return value;
    }
}
*/



float bilinear_interpolate(image im, float x, float y, int c)
{
    // TODO
    /***********************************************************************
      This function performs bilinear interpolation on image "im" given
      a floating column value "x", row value "y" and integer channel "c".
      It interpolates and returns the interpolated value.
    ************************************************************************/
    int left = (int)floor(x);
    int right = left + 1;
    int top = (int)floor(y);
    int bottom = top + 1;

    // left = clamp_int(left, 0, im.w - 1);
    // right = clamp_int(right, 0, im.w - 1);
    // top = clamp_int(top, 0, im.h - 1);
    // bottom = clamp_int(bottom, 0, im.h - 1);

    
    left = (left < 0) ? 0 : ((left >= im.w - 1) ? im.w - 1 : left);
    right = (right < 0) ? 0 : ((right >= im.w - 1) ? im.w - 1 : right);
    top = (top < 0) ? 0 : ((top >= im.h - 1) ? im.h - 1 : top);
    bottom = (bottom < 0) ? 0 : ((bottom >= im.h - 1) ? im.h - 1 : bottom);
    

    float dx = x - left;
    float dy = y - top;    

    float top_left = get_pixel(im, left, top, c);
    float top_right = get_pixel(im, right, top, c);
    float bottom_left = get_pixel(im, left, bottom, c);
    float bottom_right = get_pixel(im, right, bottom, c);

    float interpolated_value = (1 - dx) * (1 - dy) * top_left +
                                dx * (1 - dy) * top_right +
                                (1 - dx) * dy * bottom_left +
                                dx * dy * bottom_right;
    //float interpolated_value = (1 - dx) * (1 - dy) * top_left + dx * (1 - dy) * top_right + (1 - dx) * dy * bottom_left + dx * dy * bottom_right;
    return interpolated_value;
}

image bilinear_resize(image im, int w, int h)
{
    // TODO
    /***********************************************************************
      This function uses bilinear interpolation on image "im" to a new image
      of size "w x h". Algorithm is same as nearest-neighbor interpolation.
    ************************************************************************/
    image resized = make_image(w, h, im.c);

    float sx = (float)(im.w) / w;
    float sy = (float)(im.h) / h;

    for (int y = 0; y < h; y++){
      for (int x = 0; x < w; x++){
        float old_x = (x + 0.5) * sx - 0.5;
        float old_y = (y + 0.5) * sy - 0.5;

        for (int c = 0; c < im.c; c++){
          float value = bilinear_interpolate(im, old_x, old_y, c);
          set_pixel(resized, x, y, c, value);
        }
      }
    }
    return resized;
}


/********************** Filtering: Box filter ***************************
  We want to create a box filter. We will only use square box filters.
************************************************************************/

void l1_normalize(image im)
{
    // TODO
    /***********************************************************************
      This function divides each value in image "im" by the sum of all the
      values in the image and modifies the image in place.
    ************************************************************************/
    float sum[im.c];
    for (int c = 0; c < im.c; c++){
      sum[c] = 0;
      for (int i = 0; i < im.w * im.h; i++){
        sum[c] += im.data[c * im.w * im.h + i];
      }
    }

    for (int c = 0; c < im.c; c++){
      float factor = 1.0 / sum[c];
      for (int i = 0; i < im.w * im.h; i++){
        im.data[c * im.w * im.h + i] *= factor;
      }
    }
}

image make_box_filter(int w)
{
    // TODO
    /***********************************************************************
      This function makes a square filter of size "w x w". Make an image of
      width = height = w and number of channels = 1, with all entries equal
      to 1. Then use "l1_normalize" to normalize your filter.
    ************************************************************************/
    image filter = make_image(w, w, 1);

    for (int i = 0; i < w * w; i++) {
        filter.data[i] = 1;
    }

    l1_normalize(filter);
    return filter;
}

image convolve_image(image im, image filter, int preserve)
{
    // TODO
    /***********************************************************************
      This function convolves the image "im" with the "filter". The value
      of preserve is 1 if the number of input image channels need to be 
      preserved. Check the detailed algorithm given in the README.  
    ************************************************************************/
    
    assert(filter.c == im.c || filter.c == 1);
    //int output_c = preserve ? im.c : 1;

    image conv = make_image(im.w, im.h, im.c);
    image np_conv = make_image(im.w, im.h, 1);
    
    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            float np_conv_val = 0;
            for (int c = 0; c < im.c; c++) {
                float conv_val = 0;
                
                for (int fy = 0; fy < filter.h; fy++) {
                    for (int fx = 0; fx < filter.w; fx++) {
                        int ix = x + fx - (int)(filter.w / 2);
                        int iy = y + fy - (int)(filter.h / 2);
                        float filter_val = get_pixel(filter, fx, fy, filter.c == 1 ? 0 : c);
                        float image_val = get_pixel(im, ix, iy, c);
                        conv_val += filter_val * image_val;
                    }
                }
                
                if (!preserve){np_conv_val += conv_val;}
                else{set_pixel(conv, x, y, c, conv_val);}
            }
            if (!preserve) {set_pixel(np_conv, x, y, 0, np_conv_val);}
        }
    } 
    if (!preserve) {return np_conv;}
    else {return conv;}
}

image make_highpass_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 filter with highpass filter values using image.data[]
    ************************************************************************/
    image filter = make_image(3, 3, 1);

    float highpass_filter[9] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
    for (int i = 0; i < 9; i++){
      filter.data[i] = highpass_filter[i];
    }
    return filter;
}

image make_sharpen_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 filter with sharpen filter values using image.data[]
    ************************************************************************/
    image filter = make_image(3, 3, 1);

    float sharpen_filter[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    for (int i = 0; i < 9; i++){
      filter.data[i] = sharpen_filter[i];
    }
    return filter;
}

image make_emboss_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 filter with emboss filter values using image.data[]
    ************************************************************************/
    image filter = make_image(3, 3, 1);
    float emboss_filter[9] = {-2, -1, 0, -1, 1, 1, 0, 1, 2};
    for (int i = 0; i < 9; i++){
      filter.data[i] = emboss_filter[i];
    }
    return filter;
}

//------------------------------------------------------------------------
// Question 2.3.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
// Answer: TODO
/*
Highpass Filter: We shouldn't use preserve for highpass filter. Highpass filter is usually applied to the 
graytone image, so it doesn't need to preserve the color channels. It can be considered an edge detection filter. 
Since it operates on grayscale intensity values and does not modify color channels, it should not use preserve. 
We should treat the output as a single-channel image, as the result of convolution with this filter typically 
results in a grayscale image representing edges.

Sharpen Filter: We should use preserve for sharpen filter. Because sharpen Filter is used to enhance edges and 
details in images, making them appear clearer and sharper. It achieves this goal by enhancing the contrast of 
pixel values, thereby highlighting subtle details and edges in the image. Therefore, when using Sharpen filter, 
we need to retain the number of channels in the input image to achieve the sharpening effect of the original image.

Emboss Filter: We should use preserve for emboss filter.Because Emboss Filter is used to create a visual effect 
that makes an image appear as a raised or recessed three-dimensional object. It simulates the reflection effect 
of light by changing the brightness value of pixels, based on their difference from surrounding pixels, thereby 
creating a sense of stereoscopy. Therefore, when using an Emboss filter, it is necessary to maintain the number 
of channels in the original image.
*/

//------------------------------------------------------------------------

// Question 2.3.2: Do we have to do any post-processing for the above filters? Which ones and why?
// Answer: TODO
/*
It depends on the image effect you want.

Highpass Filter: Post-processing after applying a high-pass filter may involve additional steps such as 
thresholding or normalization. Thresholding can help remove noise or unwanted low-frequency components 
from the filtered image, enhancing the clarity of edges and details. Normalization techniques can be applied 
to adjust the contrast or brightness of the filtered image, ensuring that it looks visually appealing.

Sharpen Filter: After applying a sharpen filter, it's common to perform post-processing to mitigate any 
artifacts introduced by the filter, such as noise amplification or halos around edges. Techniques like 
noise reduction or edge-preserving smoothing can be applied to refine the sharpness effect and produce 
a more natural-looking result. 

Emboss Filter: Post-processing for emboss filters may involve adjustments to the intensity or direction 
of the embossing effect. Depending on the desired outcome, additional filtering or blending techniques 
can be applied to enhance the three-dimensional appearance of the image or to integrate the embossed 
effect more seamlessly with the original image content. Since emboss filters can sometimes produce 
exaggerated or unrealistic results, post-processing may also include techniques to tone down the effect 
or make it more subtle.
*/

//------------------------------------------------------------------------

image make_gaussian_filter(float sigma)
{
    // TODO
    /***********************************************************************
      sigma: a float number for the Gaussian.
      Create a Gaussian filter with the given sigma. Note that the kernel size 
      is the next highest odd integer from 6 x sigma. Return the Gaussian filter.
    ************************************************************************/
    int size = ceil(6 * sigma);
    if (size % 2 == 0) size++;
    int center = size / 2;

    image filter = make_image(size, size, 1);

    float sum = 0;
    for (int y = 0; y < size; y++){
      for (int x = 0; x < size; x++){
        float dx = x - center;
        float dy = y - center;
        float distance = dx * dx + dy * dy;

        float weight = exp(-distance / (2 * sigma * sigma)) / (TWOPI * sigma * sigma);

        set_pixel(filter, x, y, 0, weight);
        sum += weight;
      }
    }

    for (int i = 0; i < size * size; i++){
      filter.data[i] /= sum;
    }

    return filter;
}

image add_image(image a, image b)
{
    // TODO
    /***********************************************************************
      The input images a and image b have the same height, width, and channels.
      Sum the given two images and return the result, which should also have
      the same height, width, and channels as the inputs. Do necessary checks.
    ************************************************************************/
    assert(a.w == b.w && a.h == b.h && a.c == b.c);
    image result = make_image(a.w, a.h, a.c);

    for (int i = 0; i < a.w * a.h * a.c; i++){
      result.data[i] = a.data[i] + b.data[i];
    }
    return result;
}

image sub_image(image a, image b)
{
    // TODO
    /***********************************************************************
      The input image a and image b have the same height, width, and channels.
      Subtract the given two images and return the result, which should have
      the same height, width, and channels as the inputs. Do necessary checks.
    ************************************************************************/
    assert(a.w == b.w && a.h == b.h && a.c == b.c);
    image result = make_image(a.w, a.h, a.c);

    for (int i = 0; i < a.w * a.h * a.c; i++){
      result.data[i] = a.data[i] - b.data[i];
    }
    return result;
}

image make_gx_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 Sobel Gx filter and return it
    ************************************************************************/
    image filter = make_image(3, 3, 1);
    float gx_filter[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    for (int i = 0; i < 9; i++){
      filter.data[i] = gx_filter[i];
    }
    return filter;
}

image make_gy_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 Sobel Gy filter and return it
    ************************************************************************/
    image filter = make_image(3, 3, 1);
    float gy_filter[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    for (int i = 0; i < 9; i++){
      filter.data[i] = gy_filter[i];
    }
    return filter;
}

void feature_normalize(image im)
{
    // TODO
    /***********************************************************************
      Calculate minimum and maximum pixel values. Normalize the image by
      subtracting the minimum and dividing by the max-min difference.
    ************************************************************************/
   float min_val = im.data[0];
   float max_val = im.data[0];

   for (int i = 0; i < im.w * im.h * im.c; i++){
      if (im.data[i] < min_val) min_val = im.data[i];
      if (im.data[i] > max_val) max_val = im.data[i];
   }

    float range = max_val - min_val;

    if (range == 0) {
      for (int i = 0; i < im.w * im.h * im.c; i++){
        im.data[i] = 0;
      }
      return;
    }

    for (int i = 0; i < im.w * im.h * im.c; i++){
      im.data[i] = (im.data[i] - min_val) / range;
    }

    
}

image *sobel_image(image im)
{
    // TODO
    /***********************************************************************
      Apply Sobel filter to the input image "im", get the magnitude as sobelimg[0]
      and gradient as sobelimg[1], and return the result.
    ************************************************************************/
    image *sobelimg = calloc(2, sizeof(image));
    if (!sobelimg) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }

    image gx = convolve_image(im, make_gx_filter(), 0);
    image gy = convolve_image(im, make_gy_filter(), 0);
    // feature_normalize(gx);
    // feature_normalize(gy);

    sobelimg[0] = make_image(im.w, im.h, 1);
    sobelimg[1] = make_image(im.w, im.h, 1);

    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            float magnitude = sqrt(pow(get_pixel(gx, x, y, 0), 2) + pow(get_pixel(gy, x, y, 0), 2));
            float angle = atan2(get_pixel(gy, x, y, 0), get_pixel(gx, x, y, 0));
            set_pixel(sobelimg[0], x, y, 0, magnitude);
            set_pixel(sobelimg[1], x, y, 0, angle);
        }
    }

    free_image(gx);
    free_image(gy);

    return sobelimg;
}

void hsv_to_rgb_single(float hue, float saturation, float value, float *red, float *green, float *blue) {
    float p, q, t;
    int hi;

    hue *= 6;

    hi = floor(hue);
    float f = hue - hi;

    p = value * (1 - saturation);
    q = value * (1 - f * saturation);
    t = value * (1 - (1 - f) * saturation);

    switch (hi) {
        case 0:
        case 6:
            *red = value;
            *green = t;
            *blue = p;
            break;
        case 1:
            *red = q;
            *green = value;
            *blue = p;
            break;
        case 2:
            *red = p;
            *green = value;
            *blue = t;
            break;
        case 3:
            *red = p;
            *green = q;
            *blue = value;
            break;
        case 4:
            *red = t;
            *green = p;
            *blue = value;
            break;
        case 5:
            *red = value;
            *green = p;
            *blue = q;
            break;
    }
}

/*image colorize_sobel(image im)
{
  // TODO
  ***********************************************************************
    Create a colorized version of the edges in image "im" using the 
    algorithm described in the README.
  ************************************************************************
  image *sobelimg = sobel_image(im);
  image magnitude = sobelimg[0];
  image direction = sobelimg[1];

  image colorized = make_image(im.w, im.h, im.c);

  for (int i = 0; i < im.w * im.h * im.c; i++){
    float hue = direction.data[i] + pi;
    float saturation = magnitude.data[i];
    float value = magnitude.data[i];

    float red, green, blue;
    hsv_to_rgb_single(hue, saturation, value, &red, &green, &blue);

    colorized.data[i] = red;
    colorized.data[i + im.w * im.h] = green;
    colorized.data[i + 2 * im.w * im.h] = blue;
  }

  free_image(magnitude);
  free_image(direction);
  free(sobelimg);

  return colorized;  
}
*/

image colorize_sobel(image im)
{
  // TODO
  /***********************************************************************
    Create a colorized version of the edges in image "im" using the 
    algorithm described in the README.
  ************************************************************************/
  image *sobelimg = sobel_image(im);
  image magnitude = sobelimg[0];
  image direction = sobelimg[1];

  image colorized = make_image(im.w, im.h, 3);

  for (int i = 0; i < im.w * im.h; i++){
    float hue = direction.data[i] + pi;
    float saturation = magnitude.data[i];
    float value = magnitude.data[i];

    colorized.data[i] = hue;
    colorized.data[i + im.w * im.h] = saturation;
    colorized.data[i + 2 * im.w * im.h] = value;
  }

  hsv_to_rgb(colorized);

  free_image(magnitude);
  free_image(direction);
  free(sobelimg);

  return colorized;  
}

// EXTRA CREDIT: Median filter

int compare_floats(const void *a, const void *b) {
    return (*(float *)a > *(float *)b) - (*(float *)a < *(float *)b);
}


image apply_median_filter(image im, int kernel_size){
  image result = make_image(im.w, im.h, im.c);
  int half_kernel = kernel_size / 2;

  for (int y = 0; y < im.h; y++){
    for (int x = 0; x < im.w; x++){
      for (int c = 0; c < im.c; c++){
        float values[kernel_size * kernel_size];
        int idx = 0;
        for (int ky = -half_kernel; ky <= half_kernel; ky++){
          for (int kx = -half_kernel; kx <= half_kernel; kx++){
            values[idx++] = get_pixel(im, x + kx, y + ky, c);
          }
        }

        qsort(values, kernel_size * kernel_size, sizeof(float), compare_floats);
        set_pixel(result, x, y, c, values[kernel_size * kernel_size / 2]);
      }
    }
  }
  return result;
}




// SUPER EXTRA CREDIT: Bilateral filter


float gaussian(float x,float sigma){
  return (2 * pi * powf(sigma, 2.0)) * exp(-(powf(x, 2.0))/(2*powf(sigma, 2.0)));
}


image apply_bilateral_filter(image im, float sigma1, float sigma2){
    image ret = make_image(im.w, im.h, im.c);
    
    int kernel =  (int)ceilf(6 * sigma1) ;
    if(!(kernel % 2)) ++kernel;

    image gsf = make_gaussian_filter(sigma1); 

    for(int i = 0 ; i < im.w ; ++i){
      for(int j = 0 ; j < im.h ; ++j){
        for(int k = 0 ; k < im.c ; ++k){
          float sum = 0 ;

          for(int x = 0 ; x < kernel ; ++x){
            for(int y = 0 ; y < kernel ; ++y){
              float current_pix_v = get_pixel(im, i, j, k);
              float neighbor_pix_v = get_pixel(im, i+x-(int)(kernel/2), j+y-(int)(kernel/2), k);
              float cur_gs = get_pixel(gsf, x,y,0);
              float cur_gc = gaussian((current_pix_v-neighbor_pix_v), sigma2);
              sum += cur_gs * cur_gc;
            }
          }
          float new_pix_v = 0 ;

          for(int x = 0 ; x < kernel ; ++x){
            for(int y = 0 ; y < kernel ; ++y){
              float current_pix_v = get_pixel(im, i, j, k);
              float neighbor_pix_v = get_pixel(im, i+x-(kernel/2), j+y-(kernel/2), k);
              float cur_gs = get_pixel(gsf, x,y,0);
              float cur_gc = gaussian((current_pix_v-neighbor_pix_v), sigma2);
              new_pix_v += get_pixel(im, i+x-(int)(kernel/2), j+y-(int)(kernel/2), k) * cur_gs * cur_gc / sum ;
            }
          }
          set_pixel(ret, i,j,k,new_pix_v);
        }
      }
    }

    return ret;
}