#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
//#include <opencv2/opencv.hpp>
#include "image.h"

float get_pixel(image im, int x, int y, int c)
{
    // TODO Fill this in
    // clamp the coordinates to ensure they are within bounds
    x = (x < 0) ? 0 : (x >= im.w ? im.w - 1 : x);
    y = (y < 0) ? 0 : (y >= im.h ? im.h - 1 : y);
    c = (c < 0) ? 0 : (c >= im.c ? im.c - 1 : c);

    int index = c * im.w * im.h + y * im.w + x;

    return im.data[index];
}

void set_pixel(image im, int x, int y, int c, float v)
{
    // TODO Fill this in
    // check if the coordinates are within bounds
    if (x < 0 || x >= im.w || y < 0 || y >= im.h || c < 0 || c >= im.c){
        return; // do nothing
    }

    int index = c * im.w * im.h + y * im.w + x;

    im.data[index] = v;
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    // TODO Fill this in
    size_t data_size = im.w * im.h * im.c * sizeof(float);

    // use memcpy to copy the pixel data from the input image to the new image
    memcpy(copy.data, im.data, data_size);
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
    // TODO Fill this in
    for (int y = 0; y < im.h; y++){
        for (int x = 0; x < im.w; x++){
            int index = y * im.w + x;

            float red = im.data[index];
            float green = im.data[index + im.w * im.h];
            float blue = im.data[index + 2 * im.w * im.h];

            float grayscale_intensity = 0.299 * red + 0.587 * green + 0.114 * blue;

            gray.data[index] = grayscale_intensity;
        }
    }
    return gray;
}

void shift_image(image im, int c, float v)
{
    // TODO Fill this in
    if (c < 0 || c >= im.c) {
        fprintf(stderr, "Channel index out of bounds.\n");
        return;
    }

    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            int index = c * im.w * im.h + y * im.w + x;
            im.data[index] += v;
        }
    }
}

void clamp_image(image im)
{
    // TODO Fill this in
    for (int i = 0; i < im.w * im.h * im.c; i++) {
        im.data[i] = (im.data[i] < 0) ? 0 : ((im.data[i] > 1) ? 1 : im.data[i]);
    }
}


// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
    // TODO Fill this in
    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            int index = y * im.w + x;

            float red = im.data[index];
            float green = im.data[index + im.w * im.h];
            float blue = im.data[index + 2 * im.w * im.h];

            float max_vaule = fmaxf(fmaxf(red, green), blue);
            float min_value = fminf(fminf(red, green), blue);
            float delta = max_vaule - min_value;

            // Value(H)
            float value = max_vaule;

            // Saturation(S)
            float saturation = (max_vaule == 0) ? 0 : delta/max_vaule;

            // Hue(H)
            float hue;
            if (delta == 0) {
                hue = 0;
            }
            else {
                if (red == max_vaule) {
                    hue = (green - blue) / delta;
                }
                else if (green == max_vaule) {
                    hue = 2 + (blue - red) / delta;
                }
                else {
                    hue = 4 + (red - green) / delta;
                }
                hue /= 6;
                if (hue < 0) hue += 1;
            }

            // update channels
            im.data[index] = hue;
            im.data[index + im.w * im.h] = saturation;
            im.data[index + 2 * im.w * im.h] = value;

        }
    }

}

void hsv_to_rgb(image im)
{
    // TODO Fill this in
    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            int index = y * im.w + x;

            float hue = im.data[index] * 6;
            float saturation = im.data[index + im.w * im.h];
            float value = im.data[index + 2 * im.w * im.h];
            

            int hi = floor(hue);
            float f = hue - hi;
            
            float p = value * (1 - saturation);
            float q = value * (1 - f * saturation);
            float t = value * (1 - (1 - f) * saturation);

            float red, green, blue;
            if (hi == 0 || hi == 6) {
                red = value;
                green = t;
                blue = p;
            }
            else if (hi == 1) {
                red = q;
                green = value;
                blue = p;
            }
            else if (hi == 2) {
                red = p;
                green = value;
                blue = t;
            }
            else if (hi == 3) {
                red = p;
                green = q;
                blue = value;
            }
            else if (hi == 4) {
                red = t;
                green = p;
                blue = value;
            }
            else {
                // hi == 5
                red = value;
                green = p;
                blue = q;
            }

            // update channels
            im.data[index] = red;
            im.data[index + im.w * im.h] = green;
            im.data[index + 2 * im.w * im.h] = blue;

        }
    }
}


// 8.Extra Credit
void scale_image(image im, int c, float v) {
    if (c < 0 || c >= im.c) {
        fprintf(stderr, "Channel index out of bounds.\n");
        return;
    }

    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            int index = c * im.w * im.h + y * im.w + x;
            im.data[index] *= v;
        }
    }
}


// 9. Super Extra Credit
/*
void rgb_to_hcl(cv::Mat& rgb_image) {
    // RGB to CIEXYZ
    cv::Mat xyz_image;
    cv::cvtColor(rgb_image, xyz_image, cv::COLOR_RGB2XYZ);

    // CIEXYZ to CIELUV
    cv::Mat cieluv_image;
    cv::cvtColor(xyz_image, cieluv_image, cv::COLOR_XYZ2Luv);

    std::vector<cv::Mat> channels;
    cv::split(cieluv_image, channels);

    // Lightness(L)
    cv::Mat& lightness = channels[0];
    // Chroma(C)
    cv::Mat chroma;
    cv::sqrt(channels[1].mul(channels[1]) + channels[2].mul(channels[2]), chroma);
    // Hue(H)
    cv::Mat hue;
    cv::phase(channels[2], channels[1], hue, true);

    hue = (hue < 0) + (hue >= 0) * hue;

    std::vector<cv::Mat> hcl_channels = {hue, chroma, lightness};
    cv::merge(hcl_channels, rgb_image);

}

void hcl_to_rgb(cv::Mat& hcl_image) {
    std::vector<cv::Mat> channels;
    cv::split(hcl_image, channels);
    cv::Mat& hue = channels[0];
    cv::Mat& chroma = channels[1];
    cv::Mat& lightness = channels[2];

    // Hue to CIEXYZ
    cv::Mat xyz_hue;
    cv::Mat cos_hue, sin_hue;
    cv::polarToCart(chroma, hue, cos_hue, sin_hue);
    std::vector<cv::Mat> xyz_hue_channels = { cos_hue, sin_hue, lightness };
    cv::merge(xyz_hue_channels, xyz_hue);

    // CIELUV to CIEXYZ
    cv::Mat cieluv_hue;
    cv::cvtColor(xyz_hue, cieluv_hue, cv::COLOR_Luv2XYZ);

    // CIEXYZ to RGB
    cv::cvtColor(cieluv_hue, hcl_image, cv::COLOR_XYZ2RGB);
}
*/
