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
#ifndef ACV_RANSAC_H
#define ACV_RANSAC_H

#include <cmath>                 /* for pow() */
#include <cstdlib>               /* for rand() */
#include <vector>
#include <boost/noncopyable.hpp>
#include <arx/Range.h>           /* for arx::contains */

namespace acv {
// -------------------------------------------------------------------------- //
// Ransac
// -------------------------------------------------------------------------- //
  /**
   * Ransac class implements RANdom SAmple Consensus algorithm for parameter estimation of a 
   * mathematical model from a set of observed data points which contains outliers
   * (see article on wikipedia for details on Ransac algorithm, http://en.wikipedia.org/wiki/Ransac).
   *
   * In our Ransac implementation the following cost function is used:
   *
   * \f[
   * \C_2 = \sum_i{\sigma_2(e_i)}
   * \f]
   *
   * where
   *
   * \f[
   * \sigma_2 = \begin{cases}
   *   e,      &\text{if $e < T$;}\\
   *   T,      &\text{if $e >= T$.}
   * \end{cases}
   * \f]
   *
   * Here \f$e\f$ is the fitting error and \f$T\f$ is the fitting threshold. Note that this cost
   * function differs from the standard one, which uses the the following \f$\sigma\f$:
   *
   * \f[
   * \sigma = \begin{cases}
   *   0,      &\text{if $e < T$;}\\
   *   T,      &\text{if $e >= T$.}
   * \end{cases}
   * \f]
   *
   * That means that the method implemented is not really Ransac, but MSAC (google for <i> MLESAC: 
   * A New Robust Estimator with Application to Estimating Image Geometry </i> for details).
   *
   * 
   * @param Modeller                   Modeller type.
   *
   * @see RansacModellerConcept */
  template<class Modeller>
  class Ransac: public boost::noncopyable {
  public:
    typedef typename Modeller::model_type model_type;
    typedef typename Modeller::point_type point_type;

    enum {
      MIN_POINTS_FOR_FIT = Modeller::MIN_POINTS_FOR_FIT
    };

    /** Constructor. 
     *
     * @param inlierFraction           expected fraction of 'good' sample points.
     * @param targetProbability        required probability of finding a 'good' sample.
     * @param maxError                 error threshold value for determining when a point fits a model.
     * @param pointsToAcceptModel      minimal number of points for a model to be accepted.
     * @param modeller                 ransac modeller. */
    Ransac(Modeller modeller, double inlierFraction, double targetProbability, double maxError, unsigned pointsToAcceptModel):
      mModeller(modeller),
      mInlierFraction(inlierFraction), 
      mTargetProbability(targetProbability), 
      mMaxError(maxError), 
      mPointsToAcceptModel(pointsToAcceptModel) 
    {}

    /** Finds the best model fitting the given data.
     * 
     * @param points                   list of observed data points.
     * @param outInliers               (out) inliers fitting the best model.
     * @returns                        true if the best model was found, false otherwise. */
    template<class PointRandomAccessCollection, class PointOutputCollection>
    bool operator() (const PointRandomAccessCollection& points, PointOutputCollection& outInliers) {
      if(points.size() < MIN_POINTS_FOR_FIT)
        return false;

      double actualInlierFraction = mInlierFraction;
      double bestCost = std::numeric_limits<double>::max();
      std::vector<int> bestInlierIndices;

      /* Estimate the number of iterations required. */
      int requiredIterations = estimateNumberOfIterations(mTargetProbability, actualInlierFraction, MIN_POINTS_FOR_FIT, 3.0);

      /* Iterate. */
      for(int i = 0; i < requiredIterations; i++) {
        std::vector<int> inliersIndices; /* Inlier indices. */
        std::vector<point_type> modelParams;  /* Points for model creation. */

        /* Build random samples. */
        while(true) {
          /* On gcc RAND_MAX == INT_MAX, so we need to cast it to long long. */
          int index = static_cast<int>(static_cast<long long>(points.size()) * rand() / (static_cast<long long>(RAND_MAX) + 1));
          if(arx::contains(inliersIndices, index))
            continue;
          inliersIndices.push_back(index);
          modelParams.push_back(points[index]);
          if(inliersIndices.size() >= MIN_POINTS_FOR_FIT)
            break;
        }

        /* Fit model. */
        model_type model = mModeller.fit(modelParams);

        /* Cost of the current model in terms of cost function. */
        double cost = 0.0;

        /* Clear inliersIndices. Old inliers will get there on the next step. */
        inliersIndices.clear();

        /* Check all points for fit. */
        for(std::size_t i = 0; i < points.size(); i++) {
          double error = mModeller.error(points[i], model);
          if(error < mMaxError) {
            inliersIndices.push_back(i);
            cost += error;
          } else {
            cost += mMaxError;
          }
        }

        /* Test whether we can accept current model. */
        if(inliersIndices.size() < mPointsToAcceptModel)
          continue;

        /* Compare with the best one. */
        if(cost < bestCost) {
          mBestModel = model;
          bestCost = cost;
          bestInlierIndices = inliersIndices;

          /* Update requiredIterations if needed */
          double currentInlierFraction = static_cast<double>(inliersIndices.size()) / points.size();
          if(currentInlierFraction > actualInlierFraction) {
            actualInlierFraction = currentInlierFraction;
            requiredIterations = estimateNumberOfIterations(mTargetProbability, actualInlierFraction, MIN_POINTS_FOR_FIT, 3.0);
          }
        }
      }

      /* Prepare output */
      foreach(int i, bestInlierIndices)
        outInliers.insert(outInliers.end(), points[i]);
        
      return !bestInlierIndices.empty();
    }

    /** @returns                       the best model found during ransac fitting. */
    const model_type& bestModel() const {
      return mBestModel;
    }

  private:
    /**
     * Calculate the expected number of iterations required to find a good sample with probability 
     * targetProbability when a fraction of inlierFraction of the sample points are good and at 
     * least minPointsToFitModel points are required to fit a model. Add sdFactor times the 
     * standard deviation to be sure.
     *
     * It seems that the black magic formula used in this function requires some comments, 
     * so here we go. Basically, RANSAC iterations can be modelled as a random
     * process, with a probability of selecting a good subsample on the \f$k\f$'th iteration 
     * given by geometric distribution:
     *
     * \f[
     * f(k) = (1 - \omega^p)^{k - 1} * \omega^p
     * \f]
     *
     * where \f$\omega\f$ is the fraction of inliers and \f$p\f$ is the number of points 
     * required to fit the model. The expression for probability that a good subsample was selected
     * after \f$k\f$ iterations is as follows:
     *
     * \f[
     * F(k) = sum_{i=1}^k P(i) = 1 - (1 - \omega^p)^{k - 1}
     * \f]
     *
     * Note that \f$f(k)\f$ is a probability mass function, and \f$F(k)\f$ is a cumulative distribution 
     * function for geometric distribution. We can easily estimate \f$k\f$ using the formula for \f$F(k)\f$.
     * Standard deviation for geometric distribution is as follows:
     *
     * \f[
     * SD = \frac{\sqrt{1 - \omega^p}}{\omega^p}
     * \f]
     *
     * To gain additional confidence, the standard deviation multiplied by sdFactor can be added to estimated
     * number of iterations.
     *
     * @param targetProbability        Required probability to find a good sample.
     * @param inlierFraction           Fraction of the sample points which are good.
     * @param minPointsToFitModel      Minimal number of points required to fit a model.
     * @param sdFactor                 Factor to multiply the added standard deviation by.
     * @returns                        Guess for the expected number of iterations.
     */
    static int estimateNumberOfIterations(double targetProbability, double inlierFraction, unsigned int minPointsToFitModel, double sdFactor) {
      assert(targetProbability > 0 && targetProbability < 1);
      assert(inlierFraction > 0 && inlierFraction <= 1);
      assert(minPointsToFitModel > 0);

      double successProbability = pow(inlierFraction, (int) minPointsToFitModel); /* Probability of success in a single iteration. */
      return static_cast<int>(log(1 - targetProbability) / log(1 - successProbability) + sdFactor * sqrt(1 - successProbability) / successProbability) + 1;
    }

    static_assert(Modeller::MIN_POINTS_FOR_FIT > 0, "Minimal number of points for fit must be positive.");

    Modeller mModeller;
    double mInlierFraction;
    double mTargetProbability;
    double mMaxError;
    unsigned mPointsToAcceptModel;
    model_type mBestModel;
  };


// -------------------------------------------------------------------------- //
// RansacModellerConcept
// -------------------------------------------------------------------------- //
  /** Class RansacModellerConcept.
   *
   * @param Point                      Type of a single observation.
   * @param Model                      Type of a single model. */
  template<class Point, class Model> 
  class RansacModellerConcept {
  public:
    /**  Type of a single observation. */
    typedef Point point_type;

    /** Type of a single model. */
    typedef Model model_type;

    /** Default constructor. */
    RansacModellerConcept() {};

    /** Fits the model to the given set of points.
     *
     * @param points                   points to fit the model to.
     * @returns                        model fitting the given points. */
    template<class PointRandomAccessCollection>
    model_type fit(const PointRandomAccessCollection& points) const;

    /** Calculate the fitting error of a single point against the given model.
     *
     * @param point                    point to calculate error for.
     * @param model                    model to calculate error against.
     * @returns                        fitting error. */
    double error(const point_type& point, const model_type& model) const;

    enum {
      /** Minimal number of points for a successful fit. */
      MIN_POINTS_FOR_FIT = 0xDEADBEEF 
    };
  };

} // namespace acv

#endif // ACV_RANSAC_H
