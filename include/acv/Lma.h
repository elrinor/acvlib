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
#ifndef ACV_LMA_H
#define ACV_LMA_H

#include "config.h"
#include <Eigen/Dense>
#include <arx/Utility.h>  /* for arx::sqr */

namespace acv {
// -------------------------------------------------------------------------- //
// Lma
// -------------------------------------------------------------------------- //
  /** Implementation of Levenberg–Marquardt algorithm for least squares curve fitting problem.
   * See http://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm for detailed description.
   * 
   * @param Modeller                modeller type.
   * 
   * @see LmaModellerConcept */
  template<class Modeller>
  class Lma {
  public:
    typedef typename Modeller::model_type model_type;

    /** Constructor.
     * 
     * @param modeller                 lma modeller.
     * @param maxIterations            maximal number of iterations to perform.
     * @param gradMagThreshold         magnitude threshold of the gradient of the function being optimized.
     * @param stepMagThreshold         magnitude threshold of a single solution adjustment step.
     * @param errMagThreshold         magnitude threshold of the value of the function being optimized.
     * @param damping                  damping term of levenberg-marquardt algorithm. */
    Lma(Modeller modeller, int maxIterations = 512, double gradMagThreshold = 1.0e-6, double stepMagThreshold = 1.0e-6, double errMagThreshold = 1.0e-6, double damping = 10.0): 
      mMaxIterations(maxIterations),
      mSquaredGradMagThreshold(arx::sqr(gradMagThreshold)),
      mSquaredStepMagThreshold(arx::sqr(stepMagThreshold)),
      mSquaredErrMagThreshold(arx::sqr(errMagThreshold)),
      mDamping(damping),
      mModeller(modeller) {}

    /** Fits the given modeller using Levenberg-Marquardt nonlinear minimization method.
     *
     * @param params                   initial model approximation. 
     * @returns                        optimized model. */
    model_type operator()(const model_type& approximateModel) {
      int iteration = 0;
      int paramCount = mModeller.paramNumber();
      int residCount = mModeller.residualNumber();
      float dampingTerm = 1;

      /* Allocate memory for everything. */
      Eigen::VectorXd params(paramCount);                    /* Parameters. */
      Eigen::MatrixXd j(residCount, paramCount);             /* Jacobian. */
      Eigen::VectorXd x(residCount);                         /* Residuals. */
      Eigen::VectorXd grad(paramCount);                      /* Gradient. */
      Eigen::VectorXd step(paramCount);                      /* Step. */
      Eigen::VectorXd newParams(paramCount);                 /* New parameters. */
      Eigen::MatrixXd a(paramCount, paramCount);             /* Matrix for linear solver. */

      /* Initialize. */
      mModeller.parameters(approximateModel, params);
      float error = mModeller.error(params);

      /* Iterate. */
      while(true) {
        mModeller.iteration(params, j, x);

        /* Calculate gradient. */
        grad = j.transpose() * -x;
        
        /* Drop out in case gradient magnitude hits threshold. */
        if(grad.squaredNorm() < mSquaredGradMagThreshold)
          break;

        /* Inner loop - adjust dampingTerm and update current parameters approximation. */
        while(true) {
          iteration++;

          a = j.transpose() * j + Eigen::MatrixXd::Identity(paramCount, paramCount) * dampingTerm;
          a.lu().solve(grad, &step);
          newParams = params + step;

          float newError = mModeller.error(newParams);

          if(newError < error) {
            error = newError;
            params = newParams;
            dampingTerm /= mDamping;
            break;
          } else
            dampingTerm *= mDamping;

          if(iteration > mMaxIterations)
            break;
        }

        if(iteration > mMaxIterations)
          break;

        if(step.squaredNorm() < mSquaredStepMagThreshold)
          break;

        if(error < mSquaredErrMagThreshold)
          break;
      }

      return mModeller.model(params);
    }

  private:
    int mMaxIterations;
    float mSquaredGradMagThreshold;
    float mSquaredStepMagThreshold;
    float mSquaredErrMagThreshold;
    float mDamping;
    Modeller mModeller;
  };


// -------------------------------------------------------------------------- //
// LmaModellerConcept
// -------------------------------------------------------------------------- //
  template<class Model>
  class LmaModellerConcept {
  public:
    typedef Model model_type;

    /** @returns                       number of parameters. */
    int paramNumber() const;

    /** @returns                       number of residuals. */
    int residualNumber() const;

    /** Creates a model with the given parameters.
     * 
     * @param params                   parameters vector.
     * @returns                        model for given parameters. */
    model_type model(const Eigen::VectorXd& params) const;

    /** Extracts parameters from the given model.
     * 
     * @param model                    model to extract parameters from.
     * @param params                   (out) parameters vector. */
    void parameters(const model_type& model, Eigen::VectorXd& params);

    /** Calculates residual error for given parameter vector, i.e. a value of the function being optimized.
     *
     * @param params                   parameters vector.
     * @returns                        residual error. */
    float error(const Eigen::VectorXd& params) const;

    /** Calculates Jacobian and residuals vector for given parameters. 
     *
     * @param params                   parameters vector.
     * @param jacobian                 (out) jacobian for given parameters.
     * @param residuals                (out) residuals vector for given parameters. */
    void iteration(const Eigen::VectorXd& params, Eigen::MatrixXd& jacobian, Eigen::VectorXd& residuals) const;
  };

} // namespace acv

#endif // ACV_LMA_H
