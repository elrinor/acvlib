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
#ifndef ACV_OCTAVE_H
#define ACV_OCTAVE_H

#include "config.h"
#include <cmath>                /* for pow() */
#include <functional>           /* for std::minus */
#include <vector>
#include <vigra/stdimage.hxx>
#include <vigra/separableconvolution.hxx>
#include <vigra/combineimages.hxx>

namespace acv {
// -------------------------------------------------------------------------- //
// Octave
// -------------------------------------------------------------------------- //
  /**
   * Octave stores a single octave of the scale space.
   */
  template<class VigraImage>
  class Octave {
  public:
    typedef VigraImage image_type;

    Octave() {}
    
    Octave(const image_type& image, int scales, float initSigma): mScales(scales), mInitSigma(initSigma) {
      float sigmaRatio = pow(2.0f, 1.0f / scales);
      float lastSigma = initSigma;

      image_type tmp(image.width(), image.height());

      mBlur.reserve(scales + 3);
      mBlur.push_back(image);
      for(int i = 1; i < scales + 3; i++) {
        float dSigma = lastSigma * sqrt(sigmaRatio * sigmaRatio - 1.0f);

        vigra::Kernel1D<float> gaussian;
        gaussian.initGaussian(dSigma);
        separableConvolveX(srcImageRange(mBlur.back()), destImage(tmp), kernel1d(gaussian));
        mBlur.push_back(image_type());
        mBlur.back().resize(image.width(), image.height());
        separableConvolveY(srcImageRange(tmp), destImage(mBlur.back()), kernel1d(gaussian));

        lastSigma *= sigmaRatio;
      }

      for(int i = 0; i < scales + 2; i++) {
        mDogs.push_back(image_type());
        mDogs.back().resize(image.width(), image.height());
        combineTwoImages(srcImageRange(mBlur[i]), srcImage(mBlur[i + 1]), destImage(mDogs.back()), std::minus<typename image_type::value_type>());
      }
    }

    const image_type& doubleBlurredImage() const {
      return mBlur[mScales];
    }

    int scales() const { 
      return mScales; 
    }

    float initSigma() const { 
      return mInitSigma; 
    }

    const image_type& blur(int index) const { 
      return mBlur[index]; 
    }

    const image_type& dog(int index) const { 
      return mDogs[index]; 
    }

    int width() const { 
      return blur(0).width(); 
    }

    int height() const { 
      return blur(0).height(); 
    }

  private:
    int mScales;
    float mInitSigma;
    std::vector<image_type> mBlur;
    std::vector<image_type> mDogs;
  };

} // namespace acv

#endif // ACV_OCTAVE_H
