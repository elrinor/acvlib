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
#ifndef ACV_COLLAGE_LMA_MODELLER_H
#define ACV_COLLAGE_LMA_MODELLER_H

#include "config.h"
#include <vector>
#include <Eigen/Dense>
#include <arx/Foreach.h>
#include <arx/Utility.h>
#include "CollageModel.h"
#include "Match.h"

namespace acv {
// -------------------------------------------------------------------------- //
// CollageLmaModeller
// -------------------------------------------------------------------------- //
  /**
   * @see CollageRansacModeller
   */
  template<class Match>
  class CollageLmaModeller {
  public:
    typedef Match point_type; 

    typedef CollageModel model_type;

    /** Constructor.
     * 
     * @param matches                  Collection of image matches. */
    template<class MatchInputCollection>
    CollageLmaModeller(const MatchInputCollection& matches) {
      /* This is inefficient, but we don't want to push the template 
       * up to the class scope. */
      foreach(const point_type& match, matches)
        mMatches.push_back(match);
    }

    /** @returns                       number of parameters. */
    int paramNumber() const {
      return 4;
    }

    /** @returns                       number of residuals. */
    int residualNumber() const {
      return mMatches.size();
    }

    /** Creates a model with the given parameters.
     * 
     * @param params                   parameters vector.
     * @returns                        model for given parameters. */
    model_type model(const Eigen::VectorXd& params) const {
      return CollageModel(params[0], params[1], params[2], params[3]);
    }

    /** Extracts parameters from the given model.
     * 
     * @param model                    model to extract parameters from.
     * @param params                   (out) parameters vector. */
    void parameters(const model_type& model, Eigen::VectorXd& params) {
      params[0] = model.angle();
      params[1] = model.scale();
      params[2] = model.dx();
      params[3] = model.dy();
    }

    /** Calculates residual error for given parameter vector, i.e. a value of the function being optimized.
     *
     * @param params                   parameters vector.
     * @returns                        residual error. */
    double error(const Eigen::VectorXd& params) const {
      /* Get transformation. */
      Eigen::Transform2d transform = model(params);

      /* Calculate residual error. */
      double result = 0;
      foreach(const point_type& match, mMatches)
        result += (match.first() - transform * match.second()).norm();
      return result;
    }

    /** Calculates Jacobian and residuals vector for given parameters. 
     *
     * @param params                   parameters vector.
     * @param jacobian                 (out) jacobian for given parameters.
     * @param residuals                (out) residuals vector for given parameters. */
    void iteration(const Eigen::VectorXd& params, Eigen::MatrixXd& jacobian, Eigen::VectorXd& residuals) const {
      /* Get transformation. */
      CollageModel transform = model(params);
      
      /* Fill jacobian with zeros. */
      jacobian.fill(0);

      for(unsigned i = 0; i < mMatches.size(); i++) {
        const point_type& match = mMatches[i];

        /* Each residual has 4 non-zero derivatives.
         * The formula for residual is as follows: 
         *
         * r = \sqrt{a^2 + b^2}
         *
         * Here a and b are functions of parameters, therefore:
         *
         * r' = (aa' + bb') / \sqrt{a^2 + b^2} 
         *
         * Here derivative is taken by one of the parameters.
         * Let x_0 = first(0)->x(), etc, then for (x, y) vector we have:
         *
         * (a, b) = (x_0, y_0) - (x / z, y / z)
         *
         * In our case z = 1, therefore:
         *
         * (a, b) = (x_0, y_0) - (x, y)
         *
         * So the only problem left is to calculate (x, y)'.
         * It is pretty simple too, since:
         *
         * (x, y, z) = T * (x_1, y_1, 1)
         *
         * Where T is a transformation matrix. */
        
        Eigen::Vector2d xy0 = match.first();
        Eigen::Vector2d xy1 = match.second();

        Eigen::Vector3d xyz1 = Eigen::Vector3d(xy1[0], xy1[1], 1);

        Eigen::Vector2d ab = xy0 - transform * xy1;
        double r = ab.norm();

        for(int j = 0; j < 4; j++) {
          Eigen::Vector2d dab = -(transform.derivative(j) * xyz1).start<2>();
          jacobian(i, j) = (ab[0] * dab[0] + ab[1] * dab[1]) / r;
        }
        residuals[i] = r;
      }
    }

  private:
    std::vector<point_type> mMatches;
  };

} // namespace acv

#endif // ACV_COLLAGE_LMA_MODELLER_H
