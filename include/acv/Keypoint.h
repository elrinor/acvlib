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
#ifndef ACV_KEYPOINT_H
#define ACV_KEYPOINT_H

#include "config.h"
#include <iostream>
#include <string>
#include <boost/array.hpp>
#include <Eigen/Dense>
#include <arx/Foreach.h>
#include "KeypointTraits.h"

namespace acv {
// -------------------------------------------------------------------------- //
// Keypoint
// -------------------------------------------------------------------------- //
  /**
   * Keypoint class represents a single keypoint in an image. 
   */
  class Keypoint: private KeypointTraits {
  public:
    Keypoint() {}

    Keypoint(float x, float y, float scale, float angle): mX(x), mY(y), mScale(scale), mAngle(angle) {}

    float x() const { 
      return mX; 
    }

    float y() const { 
      return mY; 
    }

    Eigen::Vector2d xy() const {
      return Eigen::Vector2d(mX, mY);
    }

    float scale() const {
      return mScale; 
    }

    float angle() const { 
      return mAngle; 
    }

    typedef unsigned char value_type;
    typedef std::size_t size_type;
    enum { static_size = DESCRIPTOR_SIZE };
    typedef boost::array<value_type, static_size> container_type;

    const container_type& descriptor() const {
      return mDescriptor;
    }

    const value_type& operator[] (size_type index) const { 
      return mDescriptor[index]; 
    }

    size_type size() const {
      return mDescriptor.size();
    }

    template<class Elem, class Traits> 
    void writeTo(std::basic_ostream<Elem, Traits>& stream) const {
      stream << "KPV1 ";
      stream << mX << " " << mY << " " << mAngle << " " << mScale << " ";
      foreach(int c, mDescriptor)
        stream << c << " ";
    }

    template<class Elem, class Traits> 
    void readFrom(std::basic_istream<Elem, Traits>& stream) {
      std::string magic;
      stream >> magic;

      if(magic != "KPV1") {
        stream.setstate(std::ios_base::failbit);
        return;
      }

      float x, y, angle, scale;
      container_type descriptor;

      stream >> x >> y >> angle >> scale;
      foreach(value_type& c, descriptor) {
        int value;
        stream >> value;
        c = static_cast<value_type>(value);
      }

      if(!stream.fail()) {
        mX = x;
        mY = y;
        mAngle = angle;
        mScale = scale;

        using std::swap;
        swap(descriptor, mDescriptor);
      }
    }

  private:
    friend class Extractor;

/*    value_type& operator[] (size_type index) { 
      return mDescriptor[index]; 
    }*/

    container_type mDescriptor;
    float mX, mY, mScale, mAngle;
  };


// -------------------------------------------------------------------------- //
// KeypointPointerProxy
// -------------------------------------------------------------------------- //
  class KeypointPointerProxy {
  public:
    typedef Keypoint::value_type value_type;
    typedef Keypoint::size_type size_type;
    enum { static_size = Keypoint::static_size };

    KeypointPointerProxy(): mKeypoint(NULL) {}

    KeypointPointerProxy(Keypoint* keyPoint): mKeypoint(keyPoint) {
      assert(keyPoint != NULL);
    }

    const value_type& operator[] (size_type index) const { 
      assert(mKeypoint != NULL);

      return static_cast<const Keypoint&>(*mKeypoint)[index]; 
    }

    size_type size() const { 
      assert(mKeypoint != NULL);

      return mKeypoint->size(); 
    }

    Keypoint* keypoint() const {
      assert(mKeypoint != NULL);

      return mKeypoint;
    }

    /*operator Keypoint* () const {
      return mKeypoint;
    }*/

  private:
    Keypoint* mKeypoint;
  };


// -------------------------------------------------------------------------- //
// IO
// -------------------------------------------------------------------------- //
  template<class Elem, class Traits> 
  inline std::basic_ostream<Elem, Traits>& operator<<(std::basic_ostream<Elem, Traits>& stream, const Keypoint& keyPoint) {
    keyPoint.writeTo(stream);
    return stream;
  }

  template<class Elem, class Traits> 
  inline std::basic_istream<Elem, Traits>& operator>>(std::basic_istream<Elem, Traits>& stream, Keypoint& keyPoint) {
    keyPoint.readFrom(stream);
    return stream;
  }

} // namespace acv

#endif // ACV_KEYPOINT_H
