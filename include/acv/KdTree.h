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
#ifndef ACV_KD_TREE_H
#define ACV_KD_TREE_H

#include <cassert>
#include <functional>              /* for std::less */
#include <algorithm>               /* for std::nth_element */
#include <iterator>                /* for std::back_inserter */
#include <vector>
#include <boost/noncopyable.hpp>
#include <boost/range.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/algorithm/fill.hpp>
#include <vigra/numerictraits.hxx> /* for vigra::NumericTraits<>::Promote */
#include <arx/Memory.h>            /* for arx::classnew_allocator */
#include <arx/Utility.h>           /* for arx::sqr */
#include <arx/Foreach.h>

namespace acv {
  namespace detail {
// -------------------------------------------------------------------------- //
// KSortedList
// -------------------------------------------------------------------------- //
    /**
     * Sorted list of fixed size that stores only the smallest items.
     */
    template<class T, class Cmp = std::less<T>, class Allocator = arx::classnew_allocator<T> >
    class KSortedList: public boost::noncopyable, private Cmp {
    public:
      typedef std::vector<T, Allocator> container_type;
      typedef typename container_type::size_type size_type;
      typedef typename container_type::value_type value_type;
      typedef typename container_type::reference reference;
      typedef typename container_type::const_reference const_reference;
      typedef typename container_type::const_iterator iterator;
      typedef typename container_type::const_iterator const_iterator;

      KSortedList(size_type capacity, const Cmp& cmp): Cmp(cmp) {
        init(capacity);
      }

      KSortedList(size_type capacity) {
        init(capacity);
      }

      const_iterator begin() const {
        return mVector.begin() + mEaten;
      }

      const_iterator end() const {
        return mVector.end();
      }

      const_reference operator[] (size_type index) const {
        return mVector[mEaten + index];
      }

      const_reference back() const {
        assert(size() > 0);

        return mVector.back();
      }

      const_reference front() const {
        return operator[] (0);
      }

      size_type size() const {
        return mVector.size() - mEaten;
      }

      size_type capacity() const {
        return mVector.capacity() - mEaten - 1;
      }

      void push_back(const value_type& elem) {
        assert(mVector.capacity() > mVector.size());

        mVector.push_back(elem);
        std::inplace_merge(mVector.begin() + mEaten, mVector.end() - 1, mVector.end(), *static_cast<Cmp*>(this));
        if(mVector.size() == mVector.capacity())
          mVector.pop_back();
      }
      
      void pop_front() {
        assert(size() > 0);

        mEaten++;
      }
      
    private:
      void init(size_type capacity) {
        assert(capacity > 0);

        mVector.reserve(capacity + 1);
        mEaten = 0;
      }

      value_type& get(size_type index) {
        return mVector[mEaten + index];
      }

      container_type mVector;
      size_type mEaten;
    };

  } // namespace detail


// -------------------------------------------------------------------------- //
// KDTree
// -------------------------------------------------------------------------- //
  /**
   * KDTree class implements a k-dimensinal tree. See wikipedia article for more
   * details: http://en.wikipedia.org/wiki/Kd_tree.
   *
   * @param Point type of a point in KDTree. It is stored locally in subroutines and in KDTree, that's why it'd better be a pointer or a class with a pointer semantics.
   *   Point must be indexable, i.e. must define operator[].
   * @param dim the dimensionality of a point. It can be guessed automatically, if you define Point::static_size.
   * @param Element type of a point's element. Can be guessed automatically if Point is a pointer type or if Point::value_type is defined.
   * @param ElementPromote a type for intermediate calculations. Defaults to int for types that are smaller than int.
   */
  template<class Point, int dim = Point::static_size, class Element = typename boost::range_value<Point>::type, class ElementPromote = typename vigra::NumericTraits<Element>::Promote> 
  class KDTree: private boost::noncopyable {
  public:
    typedef std::size_t size_type;

    /** PointEntry class. */
    class PointEntry {
    public:
      PointEntry() {};

      PointEntry(const Point& elem, const ElementPromote& distSqr): mElem(elem), mDistSqr(distSqr) {};

      ElementPromote distSqr() const {
        return mDistSqr;
      }

      const Point& elem() const {
        return mElem;
      }

      bool operator< (const PointEntry& other) const {
        return mDistSqr < other.mDistSqr;
      }

    private:
      Point mElem;
      ElementPromote mDistSqr;
    };

    /**
     * Note that this method modifies points collection rearranging its items.
     *
     * @param points array of points which are used for KDTree construction. 
     */
    template<class PointRandomAccessCollection>
    KDTree(PointRandomAccessCollection& points) {
      mSize = boost::size(points);
      mRoot = mSize > 0 ? new KDTreeNode(points) : NULL;
    }

    ~KDTree() {
      if(mRoot != NULL)
        delete mRoot;
    }

    /**
     * Find an exact nearest neighbor of point in KDTree.
     * @param point a point to search a nearest neighbor for.
     * @param (out) outDistSqr a squared euclidean distance to the nearest neighbor.
     * @returns a nearest neighbor.
     */
    Point nearestNeighbour(const Point& point, ElementPromote& outDistSqr) const {
      if(mRoot == NULL)
        return Point(); /* TODO. */

      BoundingBox rect = BoundingBox::infinite();
      return mRoot->nearestNeighbour(point, &rect, std::numeric_limits<ElementPromote>::max(), outDistSqr);
    }

    /**
     * Find k exact nearest neighbors for point.
     * @param point a point to search a nearest neighbors for.
     * @param k a number of exact nearest neighbors to find.
     * @returns an ArrayList of PointEntry containing the nearest neighbors.
     */
    template<class PointEntryInsertableCollection>
    void nearestNeighbourList(const Point& point, int k, PointEntryInsertableCollection& out) const {
      if(mRoot == NULL)
        return;

      BoundingBox infiniteBox(BoundingBox::infinite());
      detail::KSortedList<PointEntry> sortedList(k);
      mRoot->nearestNeighbourList(point, infiniteBox, sortedList, std::numeric_limits<ElementPromote>::max());
      
      boost::copy(sortedList, std::back_inserter(out));
    }

    /**
     * Find k approximate nearest neighbors for point using Best-Bin-First algorithm with fixed number of iterations.
     * @param point a point to search a nearest neighbors for.
     * @param k a number of approximate nearest neighbors to find.
     * @param maxSteps a maximal number of BBF iterations.
     * @returns an ArrayList of PointEntry containing the nearest neighbors.
     */
    template<class PointEntryInsertableCollection>
    void nearestNeighbourListBBF(const Point& point, int k, unsigned int maxSteps, PointEntryInsertableCollection& out) const {
      if(mRoot == NULL)
        return;

      BBFContext context(point, k, maxSteps);
      BoundingBox infiniteBox = BoundingBox::infinite();
      mRoot->nearestNeighbourListBBF(context, infiniteBox);

      boost::copy(context.result, std::back_inserter(out));
    }

    /**
     * @returns the number of points in KDTree
     */
    size_type size() const {
      return mSize;
    }

    /**
     * @returns a good estimation for number of iterations needed in BBF search for this KDTree.
     */
    unsigned estimateGoodBBFSearchDepth() const {
      /* Black magic involved. */
      return static_cast<unsigned>(std::max(130.0, (log(static_cast<double>(size())) / log(1000.0)) * 130.0));
    }

  private:
    /** BoundingBox class represents an axis-parallel bounding box in a high-dimensional space 
     * that tracks distance to some given point. */
    class BoundingBox {
    public:
      /** Creates uninitialized bounding box. */
      BoundingBox() {}

      /** Creates infinite bounding box. */
      static BoundingBox infinite() {
        BoundingBox result;
        result.mDistSqr = 0;
        boost::fill(result.mMin, std::numeric_limits<Element>::min());
        boost::fill(result.mMax, std::numeric_limits<Element>::max());
        return result;
      };

      /** Splits this bounding box in two, updating distances to given point.
       * Left part stays in this bounding box, right part goes to target.
       *
       * @param splitDim               index of split dimension.
       * @param splitVal               coordinate of split plane.
       * @param point                  point to track distance to.
       * @param target                 (out) right part of this bounding box. */
      void splitTo(int splitDim, Element splitVal, const Point& point, BoundingBox& target) {
        assert(splitDim >= 0 && splitDim < dim);

        /* Adjust delta for the case of point lying outside of this bounding box. */
        ElementPromote distSqrDelta = 0;
        if(point[splitDim] <= mMin[splitDim])
          distSqrDelta = -arx::sqr(static_cast<ElementPromote>(mMin[splitDim]) - static_cast<ElementPromote>(point[splitDim]));
        else if(point[splitDim] >= mMax[splitDim])
          distSqrDelta = -arx::sqr(static_cast<ElementPromote>(mMax[splitDim]) - static_cast<ElementPromote>(point[splitDim]));

        /* Apply delta. */
        (point[splitDim] > splitVal ? this : &target)->mDistSqr += 
          distSqrDelta + arx::sqr(static_cast<ElementPromote>(point[splitDim]) - static_cast<ElementPromote>(splitVal));

        /* Split. */
        target = *this;
        mMax[splitDim] = splitVal;
        target.mMin[splitDim] = splitVal;
      }

      /** @returns                     squared distance to tracked point. */
      ElementPromote distSqr() {
        return mDistSqr;
      }

    private:
      Element mMin[dim], mMax[dim]; /**< This works as a [...) interval! */
      ElementPromote mDistSqr;
    };

    class KDTreeNode;

    /** BBFEntry class. */
    struct BBFEntry {
      BoundingBox* boundingBox;
      KDTreeNode* node;
      ElementPromote dist;

      BBFEntry() {}

      BBFEntry(BoundingBox* boundingBox, KDTreeNode* node, ElementPromote dist): 
        boundingBox(boundingBox), node(node), dist(dist) {}

      bool operator< (const BBFEntry& other) const {
        return dist < other.dist;
      }
    };

    /** BBFContext class. */
    class BBFContext: private boost::noncopyable {
    public:
      Point point;
      ElementPromote maxDistSqr;
      int stepsLeft;
      detail::KSortedList<PointEntry> result;
      detail::KSortedList<BBFEntry> searchList;
      std::vector<BoundingBox> boundingBoxes;

      BBFContext(const Point& point, int k, int maxSteps): point(point), maxDistSqr(std::numeric_limits<ElementPromote>::max()), stepsLeft(maxSteps), result(k), searchList(maxSteps) {
        boundingBoxes.reserve(maxSteps + 1);
      }
    };

    
    /** Single node of a kd-tree. */
    class KDTreeNode: public boost::noncopyable {
    private:
      class PointComponentLess {
      public:
        PointComponentLess(int index): mIndex(index) {}

        bool operator() (const Point& l, const Point& r) const {
          return l[mIndex] < r[mIndex];
        }

      private:
        int mIndex;
      };

    public:
      template<class PointRandomAccessCollection>
      KDTreeNode(PointRandomAccessCollection& points): mLeft(NULL), mRight(NULL) {
        typedef typename boost::range_iterator<PointRandomAccessCollection>::type iterator;
        typename boost::range_size<PointRandomAccessCollection>::type size = boost::size(points);
        
        assert(size > 0);

        mPoint = findSplitPoint(points, mSplitDim);
        if(size / 2 > 0) {
          std::pair<iterator, iterator> range = std::make_pair(boost::begin(points), boost::begin(points) + size / 2);
          mLeft  = new KDTreeNode(range); 
        }
        if(size - size / 2 - 1 > 0) {
          std::pair<iterator, iterator> range = std::make_pair(boost::begin(points) + size / 2 + 1, boost::end(points));
          mRight = new KDTreeNode(range); 
        }
      };

      ~KDTreeNode() {
        if(mRight != NULL)
          delete mRight;
        if(mLeft != NULL)
          delete mLeft;
      }

      template<class PointRandomAccessCollection>
      static const Point& placeK(PointRandomAccessCollection& points, int sortDim, size_type k) {
        std::nth_element(boost::begin(points), boost::begin(points) + k, boost::end(points), PointComponentLess(sortDim));
        return boost::begin(points)[k];
      }

      template<class PointRandomAccessCollection>
      static const Point& findSplitPoint(PointRandomAccessCollection& points, int& outSplitDim) {
        assert(boost::size(points) > 0);

        Element min[dim], max[dim];

        /* Initialize min & max. */
        boost::fill(min, std::numeric_limits<Element>::max());
        boost::fill(max, std::numeric_limits<Element>::min());
        
        /* Find min & max for each dimension. */
        foreach(const Point& point, points) {
          for(int i = 0; i < dim; i++) {
            Element val = point[i];
            if(val > max[i])
              max[i] = val;
            if(val < min[i])
              min[i] = val;
          }
        }

        /* Find 'longest' dimension. */
        int maxDiffDim;
        Element maxDiff = 0;
        for(int i = 0; i < dim; i++) {
          Element diff = max[i] - min[i];
          if(diff >= maxDiff) {
            maxDiff = diff;
            maxDiffDim = i;
          }
        }
        outSplitDim = maxDiffDim;

        return placeK(points, maxDiffDim, boost::size(points) / 2);
        /* TODO: maybe it'd be better to use middle point, not the median? */
      }

      /*Point nearestNeighbour(const Point& point, BoundingBox* leftRect, ElementPromote maxDistSqr, ElementPromote& outDistSqr) const {
        BoundingBox rightRect;
        leftRect->splitTo(mSplitDim, mPoint[mSplitDim], point, &rightRect);

        BoundingBox* nearRect;
        BoundingBox* farRect;
        KDTreeNode* nearNode;
        KDTreeNode* farNode;
        if(point[mSplitDim] <= mPoint[mSplitDim]) {
          nearRect = leftRect;
          farRect = &rightRect;
          nearNode = mLeft;
          farNode = mRight;
        } else {
          nearRect = &rightRect;
          farRect = leftRect;
          nearNode = mRight;
          farNode = mLeft;
        }

        ElementPromote distSqr;
        Point nearest;

        if(nearNode == NULL)
          distSqr = std::numeric_limits<ElementPromote>::max();
        else
          nearest = nearNode->nearestNeighbour(point, nearRect, maxDistSqr, distSqr);

        maxDistSqr = min(distSqr, maxDistSqr);

        if(farRect->distSqr() < maxDistSqr) {
          ElementPromote thisDistSqr = distanceSqr(mPoint, point);
          if(thisDistSqr < distSqr) {
            nearest = mPoint;
            distSqr = thisDistSqr;
            maxDistSqr = thisDistSqr;
          }

          ElementPromote newDistSqr;
          Point newNearest;
          if(farNode == NULL)
            newDistSqr = std::numeric_limits<ElementPromote>::max();
          else
            newNearest = farNode->nearestNeighbour(point, farRect, maxDistSqr, newDistSqr);

          if(newDistSqr < distSqr) {
            nearest = newNearest;
            distSqr = newDistSqr;
          }
        }

        outDistSqr = distSqr;
        return nearest;
      }*/

      void nearestNeighbourList(const Point& point, BoundingBox& leftRect, detail::KSortedList<PointEntry>& best, ElementPromote maxDistSqr) const {
        best.push_back(PointEntry(mPoint, distanceSqr(mPoint, point)));

        BoundingBox rightRect;
        leftRect.splitTo(mSplitDim, mPoint[mSplitDim], point, rightRect);

        BoundingBox* nearRect;
        BoundingBox* farRect;
        KDTreeNode* nearNode;
        KDTreeNode* farNode;
        if(point[mSplitDim] <= mPoint[mSplitDim]) {
          nearRect = &leftRect;
          farRect = &rightRect;
          nearNode = mLeft;
          farNode = mRight;
        } else {
          nearRect = &rightRect;
          farRect = &leftRect;
          nearNode = mRight;
          farNode = mLeft;
        }

        if(nearNode != NULL)
          nearNode->nearestNeighbourList(point, *nearRect, best, maxDistSqr);
        if(best.size() == best.capacity())
          maxDistSqr = best.back().distSqr();
        if(farNode != NULL && farRect->distSqr() < maxDistSqr)
          farNode->nearestNeighbourList(point, *farRect, best, maxDistSqr);
      }

      void nearestNeighbourListBBF(BBFContext& context, BoundingBox& leftRect) const {
        /* Save result. */
        context.result.push_back(PointEntry(mPoint, distanceSqr(mPoint, context.point)));

        if(context.stepsLeft <= 0)
          return;
        context.stepsLeft--;

        /* push_back shouldn't allocate. */
        assert(context.boundingBoxes.capacity() > context.boundingBoxes.size()); 

        context.boundingBoxes.push_back(BoundingBox());
        BoundingBox& rightRect = context.boundingBoxes.back();
        leftRect.splitTo(mSplitDim, mPoint[mSplitDim], context.point, rightRect);

        BoundingBox* nearRect;
        BoundingBox* farRect;
        KDTreeNode* nearNode;
        KDTreeNode* farNode;
        if(context.point[mSplitDim] <= mPoint[mSplitDim]) {
          nearRect = &leftRect;
          farRect = &rightRect;
          nearNode = mLeft;
          farNode = mRight;
        } else {
          nearRect = &rightRect;
          farRect = &leftRect;
          nearNode = mRight;
          farNode = mLeft;
        }

        context.searchList.push_back(BBFEntry(farRect, farNode, farRect->distSqr()));
        if(nearNode != NULL)
          nearNode->nearestNeighbourListBBF(context, *nearRect);
        if(context.result.size() == context.result.capacity())
          context.maxDistSqr = context.result.back().distSqr();
        if(context.searchList.size() > 0) {
          BBFEntry entry = context.searchList[0];
          context.searchList.pop_front();
          if(entry.node != NULL && entry.boundingBox->distSqr() < context.maxDistSqr)
            entry.node->nearestNeighbourListBBF(context, *entry.boundingBox);
        }
      }

    private:
      static ElementPromote distanceSqr(const Point& a, const Point& b) {
        ElementPromote result = 0;
        for(int i = 0; i < dim; i++)
          result += arx::sqr(static_cast<ElementPromote>(a[i]) - static_cast<ElementPromote>(b[i]));
        return result;
      }

      Point mPoint;
      int mSplitDim;
      KDTreeNode *mLeft, *mRight;
    };

    KDTreeNode *mRoot;
    size_type mSize;
  };

} // namespace acv

#endif // ACV_KD_TREE_H
