// Copyright (C) 2021 Thomas Mulvaney.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

//! Various data structures for dealing with space and geometry.

/// A point in real space
pub type Point = [f64; 3];

/// A vector in real space
pub type Vector = [f64; 3];

/// A coordinate in the voxel map
pub type Coord = [usize; 3];

/// TODO make Point, Vector and Coord concrete types so we
/// dont need to do this by making traits!
pub trait Distance {
    fn vector_to(&self, other: &Self) -> Vector;
    fn dist2(&self, other: &Self) -> f64;
    fn dist(&self, other: &Self) -> f64;
}

impl Distance for Point {
    /// Return a vector from this point to another point
    ///
    /// # Arguments
    /// * `other` - the point to calculate the vector to.
    ///
    /// # Examples
    /// ```
    /// use voxcov::geom::{Point, Distance};
    /// let a: Point = [1.0, 1.0, 1.0];
    /// let b: Point = [3.0, 4.0, 5.0];
    /// assert!(a.vector_to(&b) == [2.0, 3.0, 4.0]);
    /// ```
    #[inline]
    fn vector_to(&self, other: &Point) -> Vector {
        [other[0] - self[0], other[1] - self[1], other[2] - self[2]]
    }

    #[inline]
    fn dist2(&self, point: &Point) -> f64 {
        let v = self.vector_to(point);
        (v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2])
    }

    #[inline]
    fn dist(&self, point: &Point) -> f64 {
        f64::sqrt(self.dist2(point))
    }
}

/// Transform points from Voxel space (i, j, k) to real space (x, y, z)
#[derive(Clone, Debug, Copy)]
pub struct Transform {
    pub origin: Point,
    pub scale: Vector,
}

impl Transform {
    pub fn new(origin: Point, scale: Vector) -> Self {
        Self { origin, scale }
    }
    pub fn to_real(&self, coord: &Coord) -> Point {
        let x = (coord[0] as f64 * self.scale[0]) + self.origin[0];
        let y = (coord[1] as f64 * self.scale[1]) + self.origin[1];
        let z = (coord[2] as f64 * self.scale[2]) + self.origin[2];
        [x, y, z]
    }

    pub fn to_voxel(&self, point: &Point) -> Coord {
        let x = (point[0] - self.origin[0]) / self.scale[0];
        let y = (point[1] - self.origin[1]) / self.scale[1];
        let z = (point[2] - self.origin[2]) / self.scale[2];
        [x.round() as usize, y.round() as usize, z.round() as usize]
    }
    pub fn to_lower_voxel(&self, point: &Point) -> Coord {
        let x = (point[0] - self.origin[0]) / self.scale[0];
        let y = (point[1] - self.origin[1]) / self.scale[1];
        let z = (point[2] - self.origin[2]) / self.scale[2];
        [x.floor() as usize, y.floor() as usize, z.floor() as usize]
    }

    pub fn to_upper_voxel(&self, point: &Point) -> Coord {
        let x = (point[0] - self.origin[0]) / self.scale[0];
        let y = (point[1] - self.origin[1]) / self.scale[1];
        let z = (point[2] - self.origin[2]) / self.scale[2];
        [x.ceil() as usize, y.ceil() as usize, z.ceil() as usize]
    }
}

/// An axis aligned bounding coordinates.
#[derive(Debug)]
pub struct BoundingCoord {
    pub lower: Coord,
    pub upper: Coord,
}

/// An axis aligned bounding box (AABB).
#[derive(Debug)]
pub struct BoundingBox {
    pub lower: Point,
    pub upper: Point,
}

impl BoundingBox {
    fn nearest(&self, v: f64, lower: f64, upper: f64) -> f64 {
        // TODO maybe we should just have lower and width to define the box.
        let center = lower + ((upper - lower) / 2.0);
        if v <= center {
            lower
        } else {
            upper
        }
    }

    fn farthest(&self, v: f64, lower: f64, upper: f64) -> f64 {
        // TODO maybe we should just have lower and width to define the box.
        let center = lower + ((upper - lower) / 2.0);
        if v > center {
            lower
        } else {
            upper
        }
    }

    /// Given a point, return the nearest corner of the box.
    ///
    /// # Arguments
    /// * `point` - The point to the find the closest corner to
    ///
    /// # Examples
    /// ```
    /// use voxcov::geom::{BoundingBox, Point};
    /// let aabb = BoundingBox { lower: [0.0; 3], upper: [10.0; 3] };
    ///
    /// // A point inside the box
    /// let p: Point = [1.0, 9.0, 1.0];
    /// assert_eq!(aabb.nearest_corner(&p), [0.0, 10.0, 0.0]);
    ///
    /// // A point outside the box
    /// let p: Point = [0.0, -9.0, 0.0];
    /// assert_eq!(aabb.nearest_corner(&p), [0.0, 0.0, 0.0]);
    ///
    /// // Another point outside the box
    /// let p: Point = [11.0, 19.0, 2.0];
    /// assert_eq!(aabb.nearest_corner(&p), [10.0, 10.0, 0.0]);
    /// ```
    pub fn nearest_corner(&self, point: &Point) -> Point {
        [
            self.nearest(point[0], self.lower[0], self.upper[0]),
            self.nearest(point[1], self.lower[1], self.upper[1]),
            self.nearest(point[2], self.lower[2], self.upper[2]),
        ]
    }

    /// Given a point, return the farthest corner of the box.
    ///
    /// # Arguments
    /// * `point` - The point to the find the closest corner to
    ///
    /// # Examples
    /// ```
    /// use voxcov::geom::{BoundingBox, Point};
    /// let aabb = BoundingBox { lower: [0.0; 3], upper: [10.0; 3] };
    /// let p: Point = [1.0, 9.0, 1.0];
    /// assert_eq!(aabb.farthest_corner(&p), [10.0, 0.0, 10.0]);
    ///
    /// // A point outside the box
    /// let p: Point = [0.0, -9.0, 0.0];
    /// assert_eq!(aabb.farthest_corner(&p), [10.0, 10.0, 10.0]);
    ///
    /// // Another point outside the box
    /// let p: Point = [11.0, 19.0, 2.0];
    /// assert_eq!(aabb.farthest_corner(&p), [0.0, 0.0, 10.0]);
    /// ```
    pub fn farthest_corner(&self, point: &Point) -> Point {
        [
            self.farthest(point[0], self.lower[0], self.upper[0]),
            self.farthest(point[1], self.lower[1], self.upper[1]),
            self.farthest(point[2], self.lower[2], self.upper[2]),
        ]
    }
}

pub struct Sphere {
    pub point: Point,
    pub radius: f64,
}

impl Sphere {
    /// Create a sphere.
    ///
    /// # Arguments
    ///
    /// * `point` - The center of the sphere.
    /// * `radius` - The radius of the sphere.
    pub fn new(point: Point, radius: f64) -> Sphere {
        Sphere { point, radius }
    }

    pub fn bounding_box(&self) -> BoundingBox {
        let radius = self.radius;
        let point = self.point;
        let lower = [point[0] - radius, point[1] - radius, point[2] - radius];
        let upper = [point[0] + radius, point[1] + radius, point[2] + radius];
        BoundingBox { lower, upper }
    }
}
