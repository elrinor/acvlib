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
#ifndef ACV_EXTRACT_H
#define ACV_EXTRACT_H

#include "config.h"
#include <vector>
#include <boost/noncopyable.hpp>
#include <arx/Foreach.h>
#include <arx/Memory.h> /* for arx::classnew_allocator<>. */
#include "Keypoint.h"

namespace acv {
// -------------------------------------------------------------------------- //
// Extract
// -------------------------------------------------------------------------- //
  /**
   * Extract is a collection of keypoints extracted from an image.
   */
  template<class Allocator = arx::classnew_allocator<Keypoint> >
  class Extract: public boost::noncopyable {
  public:
    typedef Allocator allocator_type;
    typedef typename allocator_type::pointer pointer;

    Extract(): mWidth(0), mHeigth(0) {}

    Extract(int width, int height): mWidth(width), mHeigth(height) {}

    Extract(int width, int height, const allocator_type& allocator): mWidth(width), mHeigth(height), mAllocator(allocator) {}

    ~Extract() {
      foreach(pointer keypoint, mKeypoints) {
        mAllocator.destroy(keypoint);
        mAllocator.deallocate(keypoint, 1);
      }
    }

    void addKeypoint(const Keypoint& keypoint) {
      pointer localKeypoint = mAllocator.allocate(1);
      mAllocator.construct(localKeypoint, keypoint);
      mKeypoints.push_back(localKeypoint);
    }

    const std::vector<pointer>& keypoints() const {
      return mKeypoints;
    }

    const pointer keypoint(int index) const {
      return mKeypoints[index];
    }

    int width() const {
      return mWidth;
    }

    int height() const {
      return mHeigth;
    }

    void setWidth(int width) {
      mWidth = width;
    }

    void setHeight(int height) {
      mHeigth = height;
    }

    template<class Elem, class Traits> 
    friend inline std::basic_ostream<Elem, Traits>& operator<<(std::basic_ostream<Elem, Traits>& stream, const Extract& extract) {
      stream << "ARXKPF2 " << extract.mWidth << " " << extract.mHeigth << " " << extract.mKeypoints.size() << std::endl;
      foreach(pointer keypoint, extract.mKeypoints)
        stream << *keypoint << std::endl;
      return stream;
    }

    template<class Elem, class Traits> 
    friend inline std::basic_istream<Elem, Traits>& operator>>(std::basic_istream<Elem, Traits>& stream, Extract& extract) {
      int width, height, size;
      std::string magic;

      stream >> magic >> width >> height >> size;
      if(stream.bad() || magic != "ARXKPF2" || width <= 0 || height <= 0 || size < 0) {
        stream.setstate(std::ios_base::failbit);
        return stream;
      }

      std::vector<pointer> keypoints;
      for(int i = 0; i < size; i++) {
        pointer keypoint = extract.mAllocator.allocate(1);
        extract.mAllocator.construct(keypoint, Keypoint());
        stream >> *keypoint;
        keypoints.push_back(keypoint);

        if(stream.fail()) {
          foreach(pointer keypoint, keypoints) {
            extract.mAllocator.destroy(keypoint);
            extract.mAllocator.deallocate(keypoint, 1);
          }
          keypoints.clear();

          break;
        }
      }

      if(!stream.fail()) {
        extract.mWidth = width;
        extract.mHeigth = height;

        using std::swap;
        swap(keypoints, extract.mKeypoints);
      }

      return stream;
    }

  private:
    friend class Extractor;

    allocator_type mAllocator;
    int mWidth, mHeigth;
    std::vector<pointer> mKeypoints;
  };


} // namespace acv

#endif // ACV_EXTRACT_H
