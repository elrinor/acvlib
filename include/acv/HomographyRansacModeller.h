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
#ifndef ACV_HOMOGRAPHY_RANSAC_MODELLER_H
#define ACV_HOMOGRAPHY_RANSAC_MODELLER_H

#include "config.h"
#include <cassert>
#include <vector>
#include <Eigen/Dense>
#include <arx/StaticBlock.h>
#include "Match.h"

namespace acv {
// -------------------------------------------------------------------------- //
// HomographyRansacModeller
// -------------------------------------------------------------------------- //
  /** 
   * HomographyRansacModeller implements a RansacModellerConcept concept and models a generic 2D affine transformation, i.e.
   * an affine transformation with a matrix of the following form:
   * <pre>
   * a b c
   * d e f
   * g h 1
   * </pre>
   *
   * Some info on the method used for transformation reconstruction is given here:
   * http://alumni.media.mit.edu/~cwren/interpolator/
   */
  template<class Match>
  class HomographyRansacModeller {
  public:
    typedef Match point_type;

    /** Transformation that transforms coordinates in the second image into 
     * coordinates in the first is used as a model. */
    typedef Eigen::Transform2d model_type;

    enum {
      /** Minimal number of points for a successful fit. */
      MIN_POINTS_FOR_FIT = 4 
    };

    /** Constructor.
     * 
     * @param squaredScale             Area of the first image to be matched. */
    HomographyRansacModeller(double squaredScale): 
      mSquaredScale(squaredScale) {};

    /** Constructor.
     * 
     * @param width                    Width of the first image to be matched.
     * @param height                   Height of the first image to be matched. */
    HomographyRansacModeller(int width, int height): 
      mSquaredScale(static_cast<double>(width) * height) {}


    /** Fits the model to the given set of points.
     *
     * @param matches                  matches to fit the model to.
     * @returns                        model fitting the given matches. */
    template<class PointRandomAccessCollection>
    model_type fit(const PointRandomAccessCollection& matches) const {
      assert(matches.size() >= MIN_POINTS_FOR_FIT);

      /* Visit the link provided in class description for detailed explanation of
       * how it works. */

      /* Create linear system for calculation of transformation matrix coefficients. */
      Eigen::Matrix<double, 8, 8> a = Eigen::Matrix<double, 8, 8>::Zero();
      Eigen::Matrix<double, 8, 1> b;
      for(int i = 0; i < 4; i++) {
        const point_type& m = matches[i];
        double x0 = m.first().x();
        double y0 = m.first().y();
        double x1 = m.second().x();
        double y1 = m.second().y();
        int k = 2 * i;
        int l = 2 * i + 1;

        a(k, 0) = a(l, 3) = x1;
        a(k, 1) = a(l, 4) = y1;
        a(k, 2) = a(l, 5) = 1;
        
        a(k, 6) = - x0 * x1;
        a(k, 7) = - x0 * y1;
        a(l, 6) = - y0 * x1;
        a(l, 7) = - y0 * y1;

        b[k] = x0;
        b[l] = y0;
      }

      /* Solve it. */
      Eigen::Matrix<double, 8, 1> f;
      a.lu().solve(b, &f);
      
      /* Wrap the solution into 2d transformation matrix. */
      Eigen::Transform2d result;
      result(0, 0) = f[0];
      result(0, 1) = f[1];
      result(0, 2) = f[2];
      result(1, 0) = f[3];
      result(1, 1) = f[4];
      result(1, 2) = f[5];
      result(2, 0) = f[6];
      result(2, 1) = f[7];
      result(2, 2) = 1;
      return result;
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
  ARX_STATIC_BLOCK(HOMOGRAPHY_RANSAC_MODELLER_H) {
    std::vector<Match> v;
    v.push_back(Match(new Keypoint(1, 1, 0, 0),  new Keypoint(0, 0, 0, 0), 0));
    v.push_back(Match(new Keypoint(1, 5, 0, 0),  new Keypoint(3, 1, 0, 0), 0));
    v.push_back(Match(new Keypoint(-1, 3, 0, 0), new Keypoint(1, 3, 0, 0), 0));
    v.push_back(Match(new Keypoint(3, 5, 0, 0),  new Keypoint(6, 1, 0, 0), 0));
    HomographyRansacModeller<Match> m(10, 10);
    HomographyRansacModeller<Match>::model_type model = m.fit(v);
    assert((model * Eigen::Vector2d(0, 0)).isApprox(Eigen::Vector2d(1, 1)));
    assert((model * Eigen::Vector2d(3, 1)).isApprox(Eigen::Vector2d(1, 5)));
    assert((model * Eigen::Vector2d(1, 3)).isApprox(Eigen::Vector2d(-1, 3)));
    assert((model * Eigen::Vector2d(6, 1)).isApprox(Eigen::Vector2d(3, 5)));
  }
#endif

} // namespace acv

#endif // ACV_HOMOGRAPHY_RANSAC_MODELLER_H
