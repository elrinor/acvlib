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
 * License along with ACVLib. If not, see <http://www.gnu.org/licenses/>. */
#ifndef ACV_KEY_POINT_TRAITS_H
#define ACV_KEY_POINT_TRAITS_H

#include "config.h"

namespace acv {
  namespace detail {
// -------------------------------------------------------------------------- //
// KeyPointTraitsBase
// -------------------------------------------------------------------------- //
    class KeyPointTraitsBase {
    public:
      /** Double image size before keypoint search? */
      static const bool DOUBLE_IMAGE_SIZE = false;

      /** Peaks in the difference-of-Gaussian function must be at least PEAK_BORDER_DISTANCE 
       * samples away from the image border. Keypoints close to the border (< ~15 pixels) will have 
       * part of the descriptor window lying outside the image and will be unstable. */
      static const int PEAK_BORDER_DISTANCE = 5;

      /** Number of discrete smoothing levels within each octave. Higher values result in more keypoints. */
      static const int OCTAVE_SCALES = 3;

      /** Maximal number of move iterations during keypoint localization */
      static const int MAX_LOCALIZE_STEPS = 5;

      /** ORIENTATION_HISTOGRAM_BINS gives the number of bins in the histogram used to determine keypoint orientation. */
      static const int ORIENTATION_HISTOGRAM_BINS = 36;

      /** These constants specify the size of the index vectors that provide
       * a descriptor for each keypoint.  The region around the keypoint is
       * sampled at INDEX_ORIENTATION_SIZE orientations with INDEX_SIZE by INDEX_SIZE bins.
       * DESCRIPTOR_SIZE is the length of the resulting index vector. */
      static const int INDEX_ORIENTATION_SIZE = 8;
      static const int INDEX_SIZE = 4;
      static const int DESCRIPTOR_SIZE = INDEX_SIZE * INDEX_SIZE * INDEX_ORIENTATION_SIZE;
    };


// -------------------------------------------------------------------------- //
// GenericKeyPointTraits
// -------------------------------------------------------------------------- //
    template<class Tag>
    class GenericKeyPointTraits: public KeyPointTraitsBase {
    public:
      /** Initial smoothing level applied before any other processing. 
       * Good values determined experimentally are in the range 1.4 to 1.8. */
      static float INITIAL_SIGMA;

      /** Magnitude of difference-of-Gaussian value at a keypoint must be above 
       * PEAK_THRESHOLD_INIT. This avoids considering low-contrast keypoints that
       * are dominated by noise. Values down to 0.02 are feasible if detection
       * of keypoints in low-contrast regions is desired. */
      static float PEAK_THRESHOLD_INIT;

      /** PEAK_THRESH_INIT is divided by OCTAVE_SCALES because more scales 
       * result in smaller difference-of-Gaussian values. */
      static float PEAK_THRESHOLD;

      /** EDGE_EIGEN_RATIO is used to eliminate keypoints that lie on an edge in the image without 
       * their position being accurately determined along the edge.  
       * This can be determined by checking the ratio of eigenvalues of a Hessian matrix 
       * of the DOG function computed at keypoint position.
       * The eigenvalues are proportional to the two principle curvatures.
       * An EDGE_EIGEN_RATIO of 10 means that all keypoints with 
       * a ratio of principle curvatures greater than 10 are discarded. */
      static float EDGE_EIGEN_RATIO;

      /** Multiple of standard deviation of Gaussian used in orientation 
       * histogram construction. */
      static float ORIENTATION_SIGMA;

      /** All local peaks in the orientation histogram are used to generate
       * keypoints as long as the local peak is within ORIENTATION_HISTOGRAM_THRESHOLD of
       * the maximum peak.  A value of 1.0 only selects a single orientation
       * at each location. */
      static float ORIENTATION_HISTOGRAM_THRESHOLD;

      /** This constant specifies how large a region is covered by each index
       * vector bin.  It gives the spacing of index samples in terms of
       * pixels at this scale (which is then multiplied by the scale of a
       * keypoint).  It should be set experimentally to as small a value as
       * possible to keep features local (good values are in range 3 to 5). */
      static float INDEX_SPACING;

      /** Width of Gaussian weighting window for index vector values.  It is
       * given relative to half-width of index, so value of 1.0 means that
       * weight has fallen to about half near corners of index patch.  A
       * value of 1.0 works slightly better than large values (which are
       * equivalent to not using weighting).  Value of 0.5 is considerably
       * worse. */
      static float INDEX_SIGMA;

      /** Index values are thresholded at this value so that regions with
       * high gradients do not need to match precisely in magnitude.
       * Best value should be determined experimentally.  Value of 1.0
       * has no effect.  Value of 0.2 is significantly better. */
      static float MAX_INDEX_VALUE;

    private:
      typedef Tag tag_type;
    };

#define ACV_CONST(name, value)                                                  \
    template<class Tag> float GenericKeyPointTraits<Tag>::name = value
    ACV_CONST(INITIAL_SIGMA,                   1.6f);
    ACV_CONST(PEAK_THRESHOLD_INIT,             0.04f);
    ACV_CONST(PEAK_THRESHOLD,                  PEAK_THRESHOLD_INIT / OCTAVE_SCALES);
    ACV_CONST(EDGE_EIGEN_RATIO,                10.0f);
    ACV_CONST(ORIENTATION_SIGMA,               1.5f);
    ACV_CONST(ORIENTATION_HISTOGRAM_THRESHOLD, 0.8f);
    ACV_CONST(INDEX_SPACING,                   3.0f);
    ACV_CONST(INDEX_SIGMA,                     1.0f);
    ACV_CONST(MAX_INDEX_VALUE,                 0.2f);
#undef ACV_CONST

  } // namespace detail


// -------------------------------------------------------------------------- //
// KeypointTraits
// -------------------------------------------------------------------------- //
  class KeypointTraits: public detail::GenericKeyPointTraits<void> {};

} // namespace acv

/* These ones are just for IDE syntax highlighting... */
#if 0
#  define DOUBLE_IMAGE_SIZE
#  define PEAK_BORDER_DISTANCE
#  define OCTAVE_SCALES
#  define MAX_LOCALIZE_STEPS
#  define ORIENTATION_HISTOGRAM_BINS
#  define INDEX_ORIENTATION_SIZE
#  define INDEX_SIZE
#  define DESCRIPTOR_SIZE
#  define INITIAL_SIGMA
#  define PEAK_THRESHOLD_INIT
#  define PEAK_THRESHOLD
#  define EDGE_EIGEN_RATIO
#  define ORIENTATION_SIGMA
#  define ORIENTATION_HISTOGRAM_THRESHOLD
#  define INDEX_SPACING
#  define INDEX_SIGMA
#  define MAX_INDEX_VALUE
#  define INDEX_ORIENTATION_SIZE
#  define INDEX_SIZE
#  define DESCRIPTOR_SIZE
#endif

#endif // ACV_KEY_POINT_TRAITS_H
