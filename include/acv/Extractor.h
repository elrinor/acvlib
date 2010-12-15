/* This file is part of ACVLib, an ArX Computer Vision Library.
 *
 * Copyright (C) 2009-2010 Alexander Fokin <apfokin@gmail.com>
 *
 * ACVLib is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * ACVLib is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License 
 * for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with ACVLib. If not, see <http://www.gnu.org/licenses/>. 
 * 
 * $Id$ */
#ifndef ACV_EXTRACTOR_H
#define ACV_EXTRACTOR_H

#include "config.h"
#include <cmath>
#include <cstring>                  /* for memset */
#include <algorithm>                /* for std::min */
#include <boost/array.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/type_traits.hpp>    /* for boost::alignment_of */
#include <boost/range/algorithm/max_element.hpp>
#include <Eigen/Dense>
#include <vigra/stdimage.hxx>
#include <vigra/resizeimage.hxx>
#include <vigra/combineimages.hxx>  /* for vigra::MagnitudeFunctor, vigra::combineTwoImages */
#include <vigra/functorexpression.hxx>
#include <arx/Utility.h>            /* for arx::sqr */
#include <arx/Memory.h>             /* for ALIGN */
#include "Octave.h"
#include "Keypoint.h"
#include "KeypointTraits.h"
#include "Extract.h"

namespace acv {
// -------------------------------------------------------------------------- //
// Extractor
// -------------------------------------------------------------------------- //
  /**
   * Extractor class performs keypoint extraction.
   */
  class Extractor: public boost::noncopyable, private KeypointTraits {
  public:
    Extractor() {}

    /** Extracts keypoints from the given image.
     * 
     * @param inputImage               image to extract keypoints from.
     * @param curSigma                 estimation of smoothing of the provided image, in terms of standard deviation of gaussian filter.
     * @param pixelSize                size of pixels of provided image relative to the size of pixels of original image.
     * @param extract                  (out) collection of keypoints. */
    template<class Allocator, class VigraAllocator>
    void operator() (const vigra::BasicImage<float, VigraAllocator>& inputImage, float curSigma, float pixelSize, Extract<Allocator>& extract) {
      typedef vigra::BasicImage<float, VigraAllocator> Image;

      extract.setWidth(inputImage.width());
      extract.setHeight(inputImage.height());

      boost::scoped_ptr<Image> img(new Image(inputImage));

      ACV_DEBUG_PRINT("Extractor::operator(): curSigma = " << curSigma << ", pixelSize = " << pixelSize);

      if(DOUBLE_IMAGE_SIZE) {
        ACV_DEBUG_PRINT("Doubling image size");

        Image* resized = new Image(img->width() * 2, img->height() * 2);
        resizeImageLinearInterpolation(srcImageRange(*img), destImageRange(*resized));
        img.reset(resized);
        
        pixelSize /= 2;
        curSigma *= 2;
      }

      if(INITIAL_SIGMA > curSigma) {
        float sigma = sqrt(INITIAL_SIGMA * INITIAL_SIGMA - curSigma * curSigma);

        ACV_DEBUG_PRINT("Applying additional smoothing, sigma = " << sigma);

        Image tmp(img->width(), img->height());
        vigra::Kernel1D<float> gaussian;
        gaussian.initGaussian(sigma);
        separableConvolveX(srcImageRange(*img), destImage(tmp), kernel1d(gaussian));
        separableConvolveY(srcImageRange(tmp), destImage(*img), kernel1d(gaussian));
        curSigma = INITIAL_SIGMA;
      }

      int minsize = std::max(2 * PEAK_BORDER_DISTANCE + 2, 32);
      while(img->width() > minsize && img->height() > minsize) {
        ACV_DEBUG_PRINT("Processing next octave, curSigma = " << curSigma << ", pixelSize = " << pixelSize);

        Octave<Image> octave(*img, OCTAVE_SCALES, curSigma);
        extractKeypoints(octave, pixelSize, extract);
        img.reset(new Image(img->width() / 2, img->height() / 2));
        resizeImageNoInterpolation(srcImageRange(octave.doubleBlurredImage()), destImageRange(*img));
        pixelSize *= 2.0;
      }
    }

  private:
    typedef ALIGN(16) boost::array<boost::array<boost::array<float, INDEX_ORIENTATION_SIZE>, INDEX_SIZE>, INDEX_SIZE> index_type;
    typedef Eigen::Matrix<float, DESCRIPTOR_SIZE, 1> unwrapped_index_type;

    static_assert((sizeof(index_type) == sizeof(unwrapped_index_type)), "Index type and unwrapped index type must be of the same size for memcpy to work.");
    /* This one fails on gcc for some reason. It seems gcc loses alignment information when passing a type to a template.
     * So, instead of a static assert, we do a runtime check.
     *
     * STATIC_ASSERT((boost::alignment_of<unwrapped_index_type>::value <= boost::alignment_of<index_type>::value)); 
     */

    /**
     * Functor for vigra::gradientBasedTransform that calculates gradient's orientation.
     */
    template <class ValueType>
    class DirectionFunctor {
    public:
      typedef ValueType first_argument_type;
      typedef ValueType second_argument_type;
      typedef typename vigra::NumericTraits<ValueType>::RealPromote result_type;
     
      result_type operator()(const first_argument_type & v1, const second_argument_type & v2) const {
        /* Lowe uses minus here. We want to be (somewhat) compatible. */
        return -atan2(static_cast<result_type>(v2), static_cast<result_type>(v1));
      }
    };

    /**
     * Finds all keypoints within the given scale space octave.
     * 
     * @param octave                   scale space octave to search keypoints in.
     * @param pixelSize                size of the pixel in the given octave relative to the original picture's pixel size.
     * @param extract                  (out) collection of keypoints. */
    template<class Allocator, class Image>
    void extractKeypoints(const Octave<Image>& octave, float pixelSize, Extract<Allocator>& extract) {
      /* Mask for locations that have a keypoint. */
      vigra::BasicImage<bool> mask(octave.width(), octave.height(), false);

      ACV_DEBUG_PRINT("Extractor::extractKeypoints()");

      for(int s = 1; s <= octave.scales(); s++) {
        Image mag(octave.width(), octave.height());
        gradientBasedTransform(srcImageRange(octave.blur(s)), destImage(mag), vigra::MagnitudeFunctor<typename Image::value_type>());

        Image dir(octave.width(), octave.height());
        gradientBasedTransform(srcImageRange(octave.blur(s)), destImage(dir), DirectionFunctor<typename Image::value_type>());

        for(int y = PEAK_BORDER_DISTANCE; y < octave.height() - PEAK_BORDER_DISTANCE; y++) {
          for(int x = PEAK_BORDER_DISTANCE; x < octave.width() - PEAK_BORDER_DISTANCE; x++) {
            /* DOG magnitude must be above 0.8 * PEAK_THRESH threshold. */
            if(std::abs(octave.dog(s)(x, y)) <= 0.8 * PEAK_THRESHOLD)
              continue;

            /* Must be local extremum. */
            if(!isLocalExtremum3x3x3(octave, x, y, s))
              continue;

            ACV_DEBUG_PRINT("Found local extremum, applying edge rule...");

            /* Must be not on edge. */
            if(isOnEdge(octave, x, y, s))
              continue;

            ACV_DEBUG_PRINT("Found keypoint, localizing...");

            /* Localize peak. */
            Eigen::Vector3f pos;
            if(!localizeKeyPoint(octave, mask, x, y, s, MAX_LOCALIZE_STEPS, pos))
              continue;

            ACV_DEBUG_PRINT("Generating descriptor...");

            generateKeypoints(mag, dir, pixelSize, pos, extract);
          }
        }
      }
    }


    /** Supplementary function that determines whether there is a local 
     * minimum o maximum at a given position in a given image.
     * 
     * @param img                      image to check for minimum / maximum.
     * @param val                      value of a minimum / maximum.
     * @param x
     * @param y
     * @returns                        true if there is a maximum o minimum at the given position, false otherwise. */
    template<class PixelType, class Alloc>
    bool isLocalExtremum3x3(const vigra::BasicImage<PixelType, Alloc>& img, PixelType val, int x, int y) {
      if(val > 0.0) {
        return !(
          img(x - 1, y - 1) > val || img(x - 1, y    ) > val || img(x - 1, y + 1) > val ||
          img(x    , y - 1) > val || img(x    , y    ) > val || img(x    , y + 1) > val || 
          img(x + 1, y - 1) > val || img(x + 1, y    ) > val || img(x + 1, y + 1) > val
        );
      } else {
        return !(
          img(x - 1, y - 1) < val || img(x - 1, y    ) < val || img(x - 1, y + 1) < val ||
          img(x    , y - 1) < val || img(x    , y    ) < val || img(x    , y + 1) < val || 
          img(x + 1, y - 1) < val || img(x + 1, y    ) < val || img(x + 1, y + 1) < val
        );
      }
    }

    /**
     * @param oct                      scale space octave.
     * @param x
     * @param y
     * @param s                        position in scale space to check for minimum / maximum.
     * @returns                        true if there is a maximum o minimum of DoG function at the given position p, false otherwise. */
    template<class Image>
    bool isLocalExtremum3x3x3(const Octave<Image>& oct, int x, int y, int s) {
      float val = oct.dog(s)(x, y);
      return 
        isLocalExtremum3x3(oct.dog(s),     val, x, y) && 
        isLocalExtremum3x3(oct.dog(s - 1), val, x, y) && 
        isLocalExtremum3x3(oct.dog(s + 1), val, x, y);
    }

    /**
     * @param oct                      scale space octave.
     * @param x
     * @param y
     * @param s                        position in scale space to check.
     * @returns                        true if the given position p in scale space is too edgelike and therefore is 
     *                                 not suitable for keypoint creation, false otherwise. */
    template<class Image>
    bool isOnEdge(const Octave<Image>& octave, int x, int y, int s) {
      const Image& img = octave.dog(s);
      
      /* Compute 2x2 Hessian values from pixel differences. */
      float d00 = img(x, y + 1) + img(x, y - 1) - 2.0f * img(x, y);
      float d11 = img(x + 1, y) + img(x - 1, y) - 2.0f * img(x, y);
      float d01 = 0.25f * (img(x + 1, y + 1) - img(x - 1, y + 1) - img(x + 1, y - 1) + img(x - 1, y - 1));

      /* Compute determinant and trace of the Hessian. */
      float det = d00 * d11 - d01 * d01;
      float trace = d00 + d11;

      /* To detect an edge response, we require the ratio of smallest to largest principle 
       * curvatures of the DOG function (eigenvalues of the Hessian) to be below a threshold. */
      return det * arx::sqr(1.0f + EDGE_EIGEN_RATIO) <= EDGE_EIGEN_RATIO * arx::sqr(trace);
    }

    /**
     * Find subpixel position of a given keypoint.
     *
     * @param octave                   scale space octave in which keypoint was found.
     * @param mask                     image mask which is used to prevent keypoint duplicates.
     * @param x
     * @param y
     * @param s
     * @param remainingMoves           number of remaining interpolation steps.
     * @param pos                      (out) subpixel keypoint position.
     * @returns                        true if everything went OK, false otherwise. The return value of false means that the 
     *                                 given position is not suitable for keypoint creation and must be discarded. */
    template<class Image>
    bool localizeKeyPoint(const Octave<Image>& octave, vigra::BasicImage<bool>& mask, int x, int y, int s, int remainingMoves, Eigen::Vector3f& pos) {
      float peakValue;
      Eigen::Vector3f offset = localizePeak(octave, x, y, s, peakValue);

      /* Move to an adjacent location if quadratic interpolation
       * is larger than 0.6 units in some direction (0.6 is used instead of
       * 0.5 to avoid jumping back and forth near boundary). 
       * Moves to adjacent scales are not performed as it is seldom useful */
      int newy = y, newx = x;
      if(offset[0] > 0.6 && x < octave.width() - 3)
        newx++;
      if(offset[0] < -0.6 && x > 3)
        newx--;
      if(offset[1] > 0.6 && y < octave.height() - 3)
        newy++;
      if(offset[1] < -0.6 && y > 3)
        newy--;
      if(remainingMoves > 0 && (newy != y || newx != x)) 
        return localizeKeyPoint(octave, mask, newx, newy, s, remainingMoves - 1, pos);

      /* Keypoint is not created if interpolation still remains far outside expected limits, 
       * o if magnitude of peak value is below threshold. */
      if(std::abs(offset[0]) > 1.5 || std::abs(offset[1]) > 1.5 || std::abs(offset[2]) > 1.5 || std::abs(peakValue) < PEAK_THRESHOLD)
        return false;

      /* Check that no keypoint has been created at this location. Otherwise, mark this mask location. */
      if(mask[y][x])
        return false;
      mask[y][x] = true;

      pos[0] = x + offset[0];
      pos[1] = y + offset[1];
      pos[2] = octave.initSigma() * pow(2.0f, (s + offset[2]) / octave.scales());

      return true;
    }

    /** Fit a 3D quadratic function through the DOG function values around
     * the location p, at which a peak has been detected. 
     *
     * @param octave                 scale space octave where peak was found.
     * @param x
     * @param y
     * @param s
     * @param peakValue              (out) interpolated DOG magnitude at peak.
     * @returns                      interpolated peak position relative to the given position. */ 
    template<class Image>
    Eigen::Vector3f localizePeak(const Octave<Image>& octave, int x, int y, int s, float& peakValue) {
        const Image& dog0 = octave.dog(s - 1);
        const Image& dog1 = octave.dog(s);
        const Image& dog2 = octave.dog(s + 1);

        /* Fill in the values of the gradient from pixel differences. */
        Eigen::Vector3f g;
        g[0] = 0.5f * (dog1[y    ][x + 1] - dog1[y    ][x - 1]);
        g[1] = 0.5f * (dog1[y + 1][x    ] - dog1[y - 1][x    ]);
        g[2] = 0.5f * (dog2[y    ][x    ] - dog0[y    ][x    ]);

        /* Fill in the values of the Hessian from pixel differences. */
        Eigen::Matrix3f h;
        h(0, 0) = dog1[y    ][x - 1] - 2.0f * dog1[y][x] + dog1[y    ][x + 1];
        h(1, 1) = dog1[y - 1][x    ] - 2.0f * dog1[y][x] + dog1[y + 1][x    ];
        h(2, 2) = dog0[y    ][x    ] - 2.0f * dog1[y][x] + dog2[y    ][x    ];
        h(0, 1) = h(1, 0) = ((dog1[y + 1][x + 1] - dog1[y + 1][x - 1]) - (dog1[y - 1][x + 1] - dog1[y - 1][x - 1])) / 4.0f;
        h(0, 2) = h(2, 0) = ((dog2[y    ][x + 1] - dog2[y    ][x - 1]) - (dog0[y    ][x + 1] - dog0[y    ][x - 1])) / 4.0f;
        h(1, 2) = h(2, 1) = ((dog2[y + 1][x    ] - dog2[y - 1][x    ]) - (dog0[y + 1][x    ] - dog0[y - 1][x    ])) / 4.0f;

        Eigen::Vector3f result;
        h.lu().solve(-g, &result);
        peakValue = dog1[y][x] + 0.5f * result.dot(g);
        return result;
    }


    /** Assign zero or more orientations to given peak location and create a 
     * keypoint for each orientation.
     *
     * @param mag                      gradient magnitude image.
     * @param dir                      gradient orientation image.
     * @param pixelSize                size of the pixel in the given octave relative to the 
     *                                 original picture's pixel size.
     * @param pos                      interpolated peak position.
     * @param keys                     (in|out) collection of keypoints to add new keypoints to. */
    template<class VigraAlloc, class Allocator>
    void generateKeypoints(const vigra::BasicImage<float, VigraAlloc>& mag, const vigra::BasicImage<float, VigraAlloc>& dir, float pixelSize, const Eigen::Vector3f& pos, Extract<Allocator>& extract) {
      boost::array<float, ORIENTATION_HISTOGRAM_BINS> histogram;
      histogram.assign(0.0f);

      /* Calculate sigma and radius of the gaussian window 
       * used for orientation histogram construction. */
      float sigma = ORIENTATION_SIGMA * pos[2];
      int radius = static_cast<int>(sigma * 3.0f);

      int px = static_cast<int>(pos[0] + 0.5f);
      int py = static_cast<int>(pos[1] + 0.5f);
       
      for(int y = py - radius; y <= py + radius; y++) {
        for(int x = px - radius; x <= px + radius; x++) {
          /* Do not use last row or column, which are not valid. */
          if(y >= 0 && x >= 0 && y < mag.height() - 2 && x < mag.width() - 2) {
            float squaredDistance = arx::sqr(y - pos[1]) + arx::sqr(x - pos[0]);

            if(mag[y][x] > 0.0f && squaredDistance < arx::sqr(radius) + 0.5f) {
              float weight = exp(-squaredDistance / (2.0f * arx::sqr(sigma)));
              
              /* Direction is in range of -M_PI to M_PI. */
              float angle = dir[y][x];
              int bin = static_cast<int>(ORIENTATION_HISTOGRAM_BINS * (angle + M_PI + 0.001f) / (2.0f * M_PI));
              
              assert(bin >= 0 && bin <= ORIENTATION_HISTOGRAM_BINS);
              bin = std::min(bin, ORIENTATION_HISTOGRAM_BINS - 1);

              histogram[bin] += weight * mag[y][x];
            }
          }
        }
      }

      /* Smooth the direction histogram using a [1/3 1/3 1/3] kernel.
       * Why do we need smoothing? Consider the following situation:
       * [..., 0.2, 0.8, 0.7, 0.8, 0.2, ...]
       *                 ^ peak is here!  */
      /* TODO: what number of steps is good enought? Lowe uses 6, libsift uses 4... */
      for(int step = 0; step < 6; step++) {
        float prev, temp;
        prev = histogram[histogram.size() - 1];
        for(unsigned int i = 0; i < histogram.size() - 1; i++) {
          temp = histogram[i];
          histogram[i] = (prev + histogram[i] + histogram[i + 1]) / 3.0f;
          prev = temp;
        }
        int i = histogram.size() - 1;
        histogram[i] = (prev + histogram[i] + histogram[0]) / 3.0f;
      }

      /* Find maximum value in histogram. */
      float maxValue = *boost::max_element(histogram);

      for(unsigned i = 0; i < histogram.size(); i++) {
        /* Discard the peaks that don't fulfill the threshold. */
        if(histogram[i] < ORIENTATION_HISTOGRAM_THRESHOLD * maxValue)
          continue;

        int prev = (i == 0 ? histogram.size() - 1 : i - 1);
        int next = (i == histogram.size() - 1 ? 0 : i + 1);

        /* Check that it actually is a peak. */
        if(histogram[i] <= histogram[prev] || histogram[i] <= histogram[next])
          continue;
         
        float correction = interpolatePeak(histogram[prev], histogram[i], histogram[next]);
        float angle = 2.0f * static_cast<float>(M_PI) * (i + 0.5f + correction) / ORIENTATION_HISTOGRAM_BINS - static_cast<float>(M_PI);
        assert(angle >= -M_PI && angle <= M_PI);
           
        /* Create keypoint with this orientation. 
         * Coordinates of a keypoint must be calculated relative to the original image, 
         * therefore we need to multiply the given peak coordinates by a factor of pixelSize. */
        Keypoint keyPoint(pixelSize * pos[0], pixelSize * pos[1], pixelSize * pos[2], angle);

        /* Create descriptor for newly added keypoint. */
        generateDescriptor(keyPoint, mag, dir, pos);

        /* Store it. */
        extract.addKeypoint(keyPoint);
      }
    }

    /** Fit a parabola to the three points (-1.0 ; left), (0.0 ; middle) and
     * (1.0 ; right) and return a number in the range [-1, 1] that represents 
     * the peak location. The center value is assumed to be greater than o
     * equal to the other values if positive, o less than if negative.
     * 
     * @param left
     * @param middle
     * @param right
     * @returns                        peak location */
    float interpolatePeak(float left, float middle, float right) {
      if(middle < 0.0f) {
        left = -left; 
        middle = -middle; 
        right = -right;
      }
      assert(middle >= left && middle >= right);
      return 0.5f * (left - right) / (left - 2.0f * middle + right);
    }

    /** Create descriptor vector for the given keypoint.
     *
     * @param key                      (in|out) keypoint to create feature vector for.
     * @param mag                      gradient magnitude image.
     * @param dir                      gradient orientation image.
     * @param pos                      keypoint location in magnitude and direction images 
     *   (it differs from its location in original image given by key.getX() 
     *   and key.getY() !)
     */
    template<class VigraAlloc>
    void generateDescriptor(Keypoint& key, const vigra::BasicImage<float, VigraAlloc>& mag, const vigra::BasicImage<float, VigraAlloc>& dir, const Eigen::Vector3f& pos) {
      /* Initialize index array. */
      index_type index;
      memset(&index, 0, sizeof(index));

      int ix = (int) (pos[0] + 0.5);
      int iy = (int) (pos[1] + 0.5);
      float aSin = sin(key.angle());
      float aCos = cos(key.angle());
              
      /* The spacing of index samples in terms of pixels at this s. */
      float spacing = pos[2] * INDEX_SPACING;

      /* Radius of index sample region must extend to diagonal corner of
       * index patch plus half sample for interpolation. */
      int radius = (int) (1.414f * spacing * (INDEX_SIZE + 1) / 2.0f + 0.5f);
              
      /* Examine all points from the gradient image that could lie within the index square. */
      for(int dy = -radius; dy <= radius; dy++) {
        for(int dx = -radius; dx <= radius; dx++) {
          /* Compute absolute coordinates within image */
          int x = ix + dx;
          int y = iy + dy;

          /* Clip at image boundaries. */
          if(x < 0 || x >= mag.width() || y < 0 || y >= mag.height())
            continue;

          /* Rotate and scale. Also, make subpixel correction. Divide
           * by spacing to put in index units. */
          float dxr = ((aCos * dx - aSin * dy) - (pos[0] - ix)) / spacing;
          float dyr = ((aSin * dx + aCos * dy) - (pos[1] - iy)) / spacing;
             
          /* Compute location of sample in terms of real-valued index array
           * coordinates.  Subtract 0.5 so that ix of 1.0 means to put full
           * weightedMag on index[1]. */
          float yr = dyr + INDEX_SIZE / 2.0f - 0.5f;
          float xr = dxr + INDEX_SIZE / 2.0f - 0.5f;

          /* Test whether this sample falls within boundary of index patch. */
          if(yr <= -1.0f || yr >= INDEX_SIZE || xr <= -1.0f || xr >= INDEX_SIZE)
            continue;
          
          /* Compute magnitude weighted by a gaussian as function of radial
           * distance from center. */
          float weightedMag = mag[y][x] * exp(-(arx::sqr(dyr) + arx::sqr(dxr)) / (2.0f * arx::sqr(INDEX_SIGMA * 0.5f * INDEX_SIZE)));

          /* Subtract keypoint orientation to give orientation relative to keypoint. */
          float orientation = dir[y][x] - key.angle();
          
          /* Put orientation in range [0, 2*PI].
           * For some reason the following code 
           *
           * orientation = fmod(orientation + 2 * M_PI, 2 * M_PI);
           *
           * sometimes triggers the assertion below on gcc, so we use a dirty method... */
          while(orientation < 0)
            orientation += static_cast<float>(2 * M_PI);
          while(orientation > 2 * M_PI)
            orientation -= static_cast<float>(2 * M_PI);
          assert(orientation >= 0 && orientation <= 2 * M_PI);

          /* Modify index */
          addToIndex(index, weightedMag, orientation, xr, yr);
        }
      }

      /* Unwrap the 3D index values into 1D vector. */
      unwrapped_index_type& unwrapped = reinterpret_cast<unwrapped_index_type&>(index);

      /* We need to check alignment. See note near the definition of index_type. */
      assert((reinterpret_cast<uintptr_t>(&unwrapped) & 15) == 0);

      /* Normalize feature vector. */
      unwrapped.normalize();

      /* Now that normalization has been done, threshold elements of
       * index vector to decrease emphasis on large gradient magnitudes. */
      bool changed = false;
      for(int i = 0; i < unwrapped.size(); i++) {
        if(unwrapped[i] > MAX_INDEX_VALUE) {
          unwrapped[i] = MAX_INDEX_VALUE;
          changed = true;
        }
      }
      if(changed)
        unwrapped.normalize();

      /* Convert to integer, assuming each element of feature vector 
       * is likely to be less than 0.5. */
       for(int i = 0; i < unwrapped.size(); i++)
         key.mDescriptor[i] = static_cast<unsigned char>(std::min(255.0f, 512.0f * unwrapped[i]));
    }

    /** Increment appropriate locations in the index to incorporate
     * the image sample.
     *
     * @param index                    index array to work on.
     * @param mag                      magnitude of the sample.
     * @param ori                      orientation of the sample (in radians).
     * @param fx                       x sample coordinate in the index. 
     * @param fy                       y sample coordinate in the index. */
    void addToIndex(index_type& index, float mag, float ori, float x, float y) {
      /* Orientation location. */
      float o = INDEX_ORIENTATION_SIZE * ori / (2 * static_cast<float>(M_PI));
 
      /* Round down to next integer. */
      int ix = static_cast<int>((x >= 0.0f) ? x : x - 1.0f);
      int iy = static_cast<int>((y >= 0.0f) ? y : y - 1.0f);  
      int io = static_cast<int>((o >= 0.0f) ? o : o - 1.0f);
 
      /* Fractional part of location. */
      float fx = x - ix;
      float fy = y - iy;         
      float fo = o - io;
       
      /* Put appropriate fraction in each of 8 buckets around this point
       * in the (x,y,o) dimensions. */
      for(int cy = 0; cy < 2; cy++) {
        int yIndex = iy + cy;
        if(yIndex < 0 || yIndex >= INDEX_SIZE)
          continue;
        float yWeight = mag * ((cy == 0) ? 1.0f - fy : fy);
        for(int cx = 0; cx < 2; cx++) {
          int xIndex = ix + cx;
          if(xIndex < 0 || xIndex >= INDEX_SIZE)
            continue;
          float xWeight = yWeight * ((cx == 0) ? 1.0f - fx : fx);
          for(int co = 0; co < 2; co++) {
            int oIndex = io + co;
            if(oIndex >= INDEX_ORIENTATION_SIZE)  /* Orientation wraps around at M_PI. */
               oIndex = 0;
            index[yIndex][xIndex][oIndex] += xWeight * ((co == 0) ? 1.0f - fo : fo);
          }
        }
      }
    }

  };

} // namespace acv

#endif // ACV_EXTRACTOR_H
