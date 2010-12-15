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
#ifndef ACV_COLLAGE_MODEL_H
#define ACV_COLLAGE_MODEL_H

#include "config.h"
#include <cmath>
#include <Eigen/Dense>

namespace acv {
// -------------------------------------------------------------------------- //
// CollageModel
// -------------------------------------------------------------------------- //
  /**
   * CollageModel models a translation-rotation-scale 2D affine transformation.
   */
  class CollageModel: public Eigen::Transform2d {
  public:
    /** Default constructor. */
    CollageModel() {}

    /** Creates a collage model that transforms points a1 and b1 into a0 and b0 respectively. */
    CollageModel(const Eigen::Vector2d& a0, const Eigen::Vector2d& b0, const Eigen::Vector2d& a1, const Eigen::Vector2d& b1) {
      /* Target transformation will transform d1 into d0. */
      Eigen::Vector2d d0 = a0 - b0;
      Eigen::Vector2d d1 = a1 - b1;

      /* Rotation angle. */
      double angle = atan2(d0[1], d0[0]) - atan2(d1[1], d1[0]);
      double sinAngle = sin(angle);
      double cosAngle = cos(angle);

      /* Scale. */
      double scale = d0.norm() / d1.norm();

      /* Redirect. */
      init(angle, scale, -scale * (cosAngle * b1[0] - sinAngle * b1[1]) + b0[0], -scale * (sinAngle * b1[0] + cosAngle * b1[1]) + b0[1]);
    }

    /** Creates a collage model with the given angle, scale and translation. */
    CollageModel(double angle, double scale, double dx, double dy) {
      init(angle, scale, dx, dy);
    }

    double dx() const {
      return mDx;
    }

    double dy() const {
      return mDy;
    }

    double angle() const {
      return mAngle;
    }

    double scale() const {
      return mScale;
    }

    Eigen::Matrix3d derivative(int paramIndex) const {
      Eigen::Matrix3d result = Eigen::Matrix3d::Zero();
      double sinAngle = sin(mAngle);
      double cosAngle = cos(mAngle);

      if(paramIndex == 0) { /* Angle derivative. */
        result(0, 0) = mScale * -sinAngle;
        result(0, 1) = mScale * -cosAngle;
        result(1, 0) = mScale * cosAngle;
        result(1, 1) = mScale * -sinAngle;
      } else if(paramIndex == 1) { /* Scale derivative. */
        result(0, 0) = cosAngle;
        result(0, 1) = -sinAngle;
        result(1, 0) = sinAngle;
        result(1, 1) = cosAngle;
      } else if(paramIndex == 2) { /* Dx derivative. */
        result(0, 2) = 1;
      } else if(paramIndex == 3) { /* Dy derivative. */
        result(1, 2) = 1;
      } else {
        Unreachable();
      }

      return result;
    }

  private:
    void init(double angle, double scale, double dx, double dy) {
      mAngle = angle;
      mScale = scale;
      mDx = dx;
      mDy = dy;

      double sinAngle = sin(mAngle);
      double cosAngle = cos(mAngle);

      CollageModel& self = *this;
      self(0, 0) = mScale * cosAngle;
      self(0, 1) = mScale * -sinAngle;
      self(0, 2) = mDx;
      self(1, 0) = mScale * sinAngle;
      self(1, 1) = mScale * cosAngle;
      self(1, 2) = mDy;
      self(2, 0) = 0.0;
      self(2, 1) = 0.0;
      self(2, 2) = 1.0;
    }

    double mScale, mAngle, mDx, mDy;
  };

} // namespace acv

#endif // ACV_COLLAGE_MODEL_H
