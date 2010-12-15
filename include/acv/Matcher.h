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
#ifndef ACV_MATCHER_H
#define ACV_MATCHER_H

#include "config.h"
#include <vector>
#include <iterator> /* for std::back_inserter */
#include <boost/range/algorithm/copy.hpp>
#include "KdTree.h"
#include "Match.h"
#include "Ransac.h"

namespace acv {
// -------------------------------------------------------------------------- //
// Matcher
// -------------------------------------------------------------------------- //
  /**
   * Matcher performs matching of two sets of keypoints.
   */
  template<class RansacModeller>
  class Matcher {
  public:
    typedef typename RansacModeller::model_type model_type;

    /**
     * Constructor.
     * 
     * @param maxError                 maximal error relative to 1st image size, must be in (0, 1) range. 
     *                                 For example, a value of 0.01 means that reprojected position on the first image of 
     *                                 each keypoint from the second image must lie within 1% of image size from the 
     *                                 position of the matching keypoint from the first image.
     * @param minimumMatches           minimum number of matches required.
     * @param maximumMatches           number of best matches to keep, or zero to keep all.
     * @param modeller                 ransac modeller. */
    Matcher(RansacModeller modeller, double maxError, unsigned minMatches, unsigned maxMatches): 
      mModeller(modeller), mMaxError(maxError), mMinMatches(minMatches), mMaxMatches(maxMatches)
    {
      assert(minMatches <= maxMatches);
      assert(0 < maxError && maxError < 1);
    }

    /** Matches keypoint collections for two images. 
     * 
     * @param first                    collection of keypoints for the first image.
     * @parma second                   collection of keypoints for the second image.
     * @param out                      (out) collection of ransac-refined matches.
     * @returns                        true if everything went fine, false otherwise. */
    template<class KeyPointInputCollection1, class KeyPointInputCollection2, class MatchInsertableCollection>
    bool operator() (const KeyPointInputCollection1& first, const KeyPointInputCollection2& second, MatchInsertableCollection& out) {
      /* Wrap second keypoint collection. */      
      std::vector<KeypointPointerProxy> secondWrapped;
      foreach(Keypoint* keypoint, second)
        secondWrapped.push_back(KeypointPointerProxy(keypoint));

      /* Build KDTree for second collection. */
      typedef KDTree<KeypointPointerProxy, KeypointPointerProxy::static_size, KeypointPointerProxy::value_type> kdtree_type;
      kdtree_type kdTree(secondWrapped);

      /* Estimate search depth. */
      int searchDepth = kdTree.estimateGoodBBFSearchDepth();

      /* Match. */
      std::vector<Match> matches;
      std::vector<kdtree_type::PointEntry> bbfResult;
      foreach(Keypoint* keypoint, first) {
        bbfResult.clear();
        kdTree.nearestNeighbourListBBF(KeypointPointerProxy(keypoint), 2, searchDepth, bbfResult);

        /* Skip non-distinctive matches */
        if(bbfResult.size() > 1 && bbfResult[0].distSqr() > arx::sqr(0.8) * bbfResult[1].distSqr())
          continue;

        matches.push_back(Match(keypoint, bbfResult[0].elem().keypoint(), bbfResult[0].distSqr()));
      }

      /* Check match number */
      if(matches.size() < 4)
        return false;

      /* Apply ransac.
       * Note that model returns squared error values, therefore we need to pass squared max error to ransac. */
      std::vector<Match> filteredMatches;
      Ransac<RansacModeller> ransac(mModeller, 0.5f, 0.98, arx::sqr(mMaxError), mMinMatches);
      if(!ransac(matches, filteredMatches))
        return false;
      mBestModel = ransac.bestModel();
  
      /* Leave only maximumMatches best matches. */
      if(mMaxMatches != 0 && filteredMatches.size() > mMaxMatches) {
        std::nth_element(filteredMatches.begin(), filteredMatches.begin() + mMaxMatches - 1, filteredMatches.end(), MatchDistCmp());
        filteredMatches.erase(filteredMatches.begin() + mMaxMatches, filteredMatches.end());
      }

      /* Write result. */
      boost::copy(filteredMatches, std::back_inserter(out));
      return true;
    }

    /** @returns the best model found during ransac fitting. */
    const model_type& bestModel() const {
      return mBestModel;
    }

  private:
    RansacModeller mModeller;
    double mMaxError;
    unsigned mMinMatches, mMaxMatches;
    model_type mBestModel;
  };

} // namespace acv

#endif // ACV_MATCHER_H
