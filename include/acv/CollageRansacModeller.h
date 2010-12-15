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
#ifndef ACV_COLLAGE_RANSAC_MODELLER_H
#define ACV_COLLAGE_RANSAC_MODELLER_H

#include "config.h"
#include <cassert>
#include <vector>
#include <Eigen/Dense>
#include <arx/StaticBlock.h>
#include "CollageModel.h"
#include "Match.h"

namespace acv {
// -------------------------------------------------------------------------- //
// CollageRansacModeller
// -------------------------------------------------------------------------- //
  /** 
   * CollageRansacModeller implements a RansacModellerConcept concept and 
   * models a translation-rotation-scale 2D affine transformation.
   */
  template<class Match>
  class CollageRansacModeller {
  public:
    /** Match between two keypoints plays a role of a single observation. */
    typedef Match point_type;

    /** Transformation that transforms coordinates in the second image into 
     * coordinates in the first is used as a model. */
    typedef CollageModel model_type;

    enum {
      /** Minimal number of points for a successful fit. */
      MIN_POINTS_FOR_FIT = 2 
    };

    /** Constructor.
     * 
     * @param squaredScale             Area of the first image to be matched. */
    CollageRansacModeller(double squaredScale): 
      mSquaredScale(squaredScale) {};

    /** Constructor.
     * 
     * @param width                    Width of the first image to be matched.
     * @param height                   Height of the first image to be matched. */
    CollageRansacModeller(int width, int height): 
      mSquaredScale(static_cast<double>(width) * height) {}

    /** Fits the model to the given set of points.
     *
     * @param matches                  matches to fit the model to.
     * @returns                        model fitting the given matches. */
    template<class PointRandomAccessCollection>
    model_type fit(const PointRandomAccessCollection& matches) const {
      assert(matches.size() >= MIN_POINTS_FOR_FIT);

      return CollageModel(matches[1].first(), matches[0].first(), matches[1].second(), matches[0].second());
    }

    /**
     * Calculate the fitting error of a single point against the current model.
     *
     * @param p 
     * @returns                        Fitting error.
     */
    double error(const point_type& match, const model_type& transform) const {
      return (match.first() - transform * match.second()).squaredNorm() / mSquaredScale;
    }

  private:
    double mSquaredScale;
  };

#ifdef _DEBUG
  /* Fast testing, just in case. */
  ARX_STATIC_BLOCK(COLLAGE_RANSAC_MODELLER_H) {
    std::vector<Match> v;
    v.push_back(Match(new Keypoint(1, 1, 0, 0), new Keypoint(0, 0, 0, 0), 0));
    v.push_back(Match(new Keypoint(1, 5, 0, 0), new Keypoint(3, 1, 0, 0), 0));
    CollageRansacModeller<Match> m(10, 10);
    acv::CollageRansacModeller<Match>::model_type model = m.fit(v);

    assert((model * Eigen::Vector2d(0, 0)).isApprox(Eigen::Vector2d(1, 1)));
    assert((model * Eigen::Vector2d(3, 1)).isApprox(Eigen::Vector2d(1, 5)));
  }
#endif

} // namespace acv

#endif // ACV_COLLAGE_RANSAC_MODELLER_H
