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
#ifndef ACV_HOMOGRAPHY_LMA_MODELLER_H
#define ACV_HOMOGRAPHY_LMA_MODELLER_H

#include "config.h"
#include <vector>
#include <Eigen/Dense>
#include <arx/Foreach.h>
#include "Match.h"

namespace acv {
// -------------------------------------------------------------------------- //
// HomographyLmaModeller
// -------------------------------------------------------------------------- //
  /**
   * HomographyRansacModeller implements a LmaModellerConcept concept and 
   * models a generic 2D affine transformation.
   *
   * @see HomographyRansacModeller
   */
  template<class Match>
  class HomographyLmaModeller {
    typedef typename Match::value_type value_type;
    class LmaMatch {
    public:
      LmaMatch(const value_type& first, const value_type& second): mFirst(first), mSecond(second) {}

      const value_type& first() const {
        return mFirst;
      }

      const value_type& second() const {
        return mSecond;
      }

      void setFirst(const value_type& first) {
        mFirst = first;
      }

      void setSecond(const value_type& second) {
        mSecond = second;
      }

    private:
      value_type mFirst, mSecond;
    };

    Eigen::Transform2d internalModel(const Eigen::VectorXd& params) const {
      Eigen::Transform2d result;
      result(0, 0) = params[0];
      result(0, 1) = params[1];
      result(0, 2) = params[2];
      result(1, 0) = params[3];
      result(1, 1) = params[4];
      result(1, 2) = params[5];
      result(2, 0) = params[6];
      result(2, 1) = params[7];
      result(2, 2) = 1;
      return result;
    }

  public:
    typedef Eigen::Transform2d model_type;
    typedef Match point_type;

    /** Constructor.
     * 
     * @param matches                  Collection of image matches. */
    template<class MatchInputCollection>
    HomographyLmaModeller(const MatchInputCollection& matches) {
      /* Copy matches. */
      foreach(const point_type& match, matches)
        mMatches.push_back(LmaMatch(match.first(), match.second()));

      assert(mMatches.size() > 0);

      /* Find centroid. */
      value_type firstCentroid, secondCentroid;
      firstCentroid = secondCentroid = value_type::Zero();
      foreach(const LmaMatch& match, mMatches) {
        firstCentroid += match.first();
        secondCentroid += match.second();
      }
      firstCentroid /= matches.size();
      secondCentroid /= matches.size();

      /* Find dispersion. */
      double firstDispersion = 0, secondDispersion = 0;
      foreach(const LmaMatch& match, mMatches) {
        firstDispersion += (match.first() - firstCentroid).norm();
        secondDispersion += (match.second() - secondCentroid).norm();
      }
      firstDispersion /= mMatches.size();
      secondDispersion /= mMatches.size();

      /* Adjust matches. */
      foreach(LmaMatch& match, mMatches) {
        match.setFirst((match.first() - firstCentroid) / firstDispersion);
        match.setSecond((match.second() - secondCentroid) / secondDispersion);
      }

      /* Save transformations. */
      mPreTransform = Eigen::Scaling2d(1 / secondDispersion) * Eigen::Translation2d(-secondCentroid);
      mPastTransform = Eigen::Translation2d(firstCentroid) * Eigen::Scaling2d(firstDispersion);
    }

    /** @returns                       number of parameters. */
    int paramNumber() const {
      return 8;
    }

    /** @returns                       number of residuals. */
    int residualNumber() const {
      return mMatches.size();
    }

    /** Creates a model with the given parameters.
     * 
     * @param params                   parameters vector.
     * @returns                        model for given parameters. */
    Eigen::Transform2d model(const Eigen::VectorXd& params) const {
      return mPastTransform * internalModel(params) * mPreTransform;
    }

    /** Extracts parameters from the given model.
     * 
     * @param model                    model to extract parameters from.
     * @param params                   (out) parameters vector. */
    void parameters(const Eigen::Transform2d& externalModel, Eigen::VectorXd& params) {
      Eigen::Matrix3d model = mPastTransform.matrix().inverse() * externalModel * mPreTransform.matrix().inverse();
      params[0] = model(0, 0);
      params[1] = model(0, 1);
      params[2] = model(0, 2);
      params[3] = model(1, 0);
      params[4] = model(1, 1);
      params[5] = model(1, 2);
      params[6] = model(2, 0);
      params[7] = model(2, 1);
    }

    /** Calculates residual error for given parameter vector, i.e. a value of the function being optimized.
     *
     * @param params                   parameters vector.
     * @returns                        residual error. */
    double error(const Eigen::VectorXd& params) const {
      /* Get transformation. */
      Eigen::Transform2d transform = internalModel(params);

      /* Calculate residual error. */
      double result = 0;
      foreach(const LmaMatch& match, mMatches)
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
      Eigen::Transform2d transform = internalModel(params);
      
      /* Fill jacobian with zeros. */
      jacobian.fill(0);

      for(unsigned i = 0; i < mMatches.size(); i++) {
        const LmaMatch& match = mMatches[i];

        /* Each residual has 8 non-zero derivatives.
         * The formula for residual is as follows: 
         *
         * r = \sqrt{a^2 + b^2}
         *
         * Here a and b are functions of parameters, therefore:
         *
         * r' = (aa' + bb') / \sqrt{a^2 + b^2} 
         *
         * Here derivative is taken by one of the parameters.
         * Let x_0 = first()->x(), etc, then for (x, y) vector we have:
         *
         * (a, b) = (x_0, y_0) - (x / z, y / z)
         *
         * Therefore we can calculate (a, b)' using chain rule:
         *
         * d(a, b) / dk = d(a, b) / d(x, y, z) \times d(x, y, z) / dk
         *
         * where
         *
         * d(a, b) / d(x, y, z) = 
         *     | -1/z    0   x/z^2 |
         *     |   0   -1/z  y/z^2 |
         *
         * So the only problem left is to calculate (x, y, z)' (denoted above as d(x, y, z) / dk).
         * It is pretty simple too, since:
         *
         * (x, y, z) = T * (x_1, y_1, 1)
         *
         * Where T is a transformation matrix. */
        
        Eigen::Vector2d xy0 = match.first();
        Eigen::Vector2d xy1 = match.second();

        Eigen::Vector3d xyz1 = Eigen::Vector3d(xy1[0], xy1[1], 1);
        Eigen::Vector3d xyz = transform.matrix() * xyz1;

        Eigen::Vector2d ab = xy0 - transform * xy1;
        
        double r = ab.norm();

        Eigen::Matrix<double, 2, 3> dab_dxyz;
        dab_dxyz(0, 0) = -1 / xyz[2];
        dab_dxyz(1, 0) = 0;
        dab_dxyz(0, 1) = 0;
        dab_dxyz(1, 1) = -1 / xyz[2];
        dab_dxyz(0, 2) = xyz[0] / arx::sqr(xyz[2]);
        dab_dxyz(1, 2) = xyz[1] / arx::sqr(xyz[2]);

        for(int j = 0; j < 8; j++) {
          Eigen::Matrix3d derivative;
          derivative.fill(0);
          derivative(j / 3, j % 3) = 1;

          Eigen::Vector3d dxyz = derivative * xyz1;
          Eigen::Vector2d dab = dab_dxyz * dxyz;
          jacobian(i, j) = (ab[0] * dab[0] + ab[1] * dab[1]) / r;
        }
        residuals[i] = r;
      }
    }

  private:
    std::vector<LmaMatch> mMatches;
    Eigen::Transform2d mPreTransform, mPastTransform;
  };

} // namespace acv

#endif // ACV_HOMOGRAPHY_LMA_MODELLER_H
