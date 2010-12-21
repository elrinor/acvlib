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
#ifndef ACV_MATCH_H
#define ACV_MATCH_H

#include "config.h"
#include <cassert>
#include <boost/array.hpp>
#include "Keypoint.h"

namespace acv {
// -------------------------------------------------------------------------- //
// Match
// -------------------------------------------------------------------------- //
  /**
   * Match class represents a match between two keypoints.
   */
  class Match {
  public:
    typedef Eigen::Vector2d value_type;

    Match() {}

    Match(Keypoint* key0, Keypoint* key1, float distSqr): mDistSqr(distSqr) {
      assert(key0 != NULL && key1 != NULL);
      assert(distSqr >= 0);

      mKeyPoints[0] = key0;
      mKeyPoints[1] = key1;
    }
    
    Eigen::Vector2d first() const {
      return mKeyPoints[0]->xy(); 
    }

    Eigen::Vector2d second() const {
      return mKeyPoints[1]->xy();
    }

    Keypoint *firstKeypoint() const {
      return mKeyPoints[0];
    }

    Keypoint *secondKeypoint() const {
      return mKeyPoints[1];
    }

    float distSqr() const {
      return mDistSqr;
    }

  private:
    boost::array<Keypoint*, 2> mKeyPoints;
    float mDistSqr;
  };


// -------------------------------------------------------------------------- //
// MatchDistCmp
// -------------------------------------------------------------------------- //
  class MatchDistCmp: public std::binary_function<Match, Match, bool> {
  public:
    bool operator()(const Match& l, const Match& r) const {
      return l.distSqr() < r.distSqr();
    }
  };


} // namespace acv

#endif // ACV_MATCH_H
