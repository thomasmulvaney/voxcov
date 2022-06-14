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

//! Various data structures for calculating voxel coverage.
//!
//! # Defining voxel coverage
//!
//! We use the same definition as in TEMPy currently.
//!
//! A voxel can be described as coordinates and a transformation.
//! The coordinates are an integer tuple (I, I, I) and the transformation
//! is an origin plus scale both of which are real numbers (R,R,R).
//!
//! When determining whether a voxel is covered, the TEMPy approach is
//! to check if a point at the center of the voxel is covered (inside
//! the sphere).  In this case the center is defined as being:
//!
//! ```text
//!   center = (Coord + [0.5, 0.5, 0.5]) * Scale + Translation
//! ```
//!
//! In other words, a map with cell size of (2.2, 2.2, 2.2) and an origin of (0, 0, 0)
//! the voxel with coordinates (3,3,3) would have have a center at (7.7, 7.7, 7.7).
//!
//! Note, that this definition of the center is different to the definition used by
//! TEMPy when blurring which maps, which defines the center as:
//!
//! ```text
//!   center = Coord * Scale + Translation
//! ```
//!
//! In other words, a map with cell size of (2.2, 2.2, 2.2) and an origin of (0, 0, 0)
//! the voxel with coordinates (3,3,3) would have have a center at (6.6, 6.6, 6.6).
//!
//! # Algorithm
//!
//! We define a map as being by recursively partitioning a cube into smaller and smaller
//! octants until the leaves are the size of voxels.
//! Because maps aren't always cubic, the starting cube must have a width equal to the
//! longest dimension of the real map.  This does not incur a space penalty as voxels are not
//! reified until covered.
//!
//! When a sphere is added to a map we determine how an octant is covered.  If it is
//! not covered, we skip it. If it is fully covered we 'record' this.  If it is partially
//! covered, we recurse, applying the same steps on the children octants.
//! How we record the coverage depends on the implementation as outlined below.
//!
//! We extend the notion of voxel coverage to the groups of voxels in each quandant.  In
//! this case a quadrant is fully covered if all voxels in the quadrant are covered, partially
//! if some of the voxels are covered and not covered if no voxels are covered.
//! This can be determined trivially by defining a bounding box where each corner is the center
//! of the eight voxel extremes of the octant and using some simple geometric checks,
//! avoiding checking every voxel in the octant.
//!
//! ## Coverage Map
//!
//! When a voxel is covered we set a flag to true.  If all the sibling octants
//! are now covered, the parent becomes covered, and children are deleted to save space.
//! This also makes future covering operations faster as larger and larger quadrants
//! become covered.
//!
//! ## Differential coverage Map
//!
//! When a voxel is covered a counter is incremented. This allows removing spheres
//! by decrementing the counter.  There is no procedure for compacting the tree
//! when all children are covered as the information is required to make removing
//! spheres possible (we are effectivle counting partial and total coverage for every node).

use crate::geom::{BoundingBox, Coord, Point, Sphere, Transform, Vector};
use crate::node::{CubeCounter, Node};
use std::collections::HashMap;
/// A volume may cover a region of voxels partially, totally or not at all.
#[derive(PartialEq)]
pub enum CoverageType {
    None,
    Partial,
    Total,
    Inside,
}

#[derive(Clone)]
pub struct MapRoot<C> {
    pub root: Node<C>,
    transform: Transform,
}

/// All coverage maps implement this trait to allow initialisation.
pub trait MapLike {
    fn new(scale: Vector, origin: Point, width: usize) -> Self;
    fn zero(&mut self);
}

pub trait AddCoverage {
    fn add_cov<B: BoxCoverage + Radius>(&mut self, obj: &B, f: &mut dyn FnMut([usize; 3]))
        -> usize;
}

pub trait DelCoverage {
    fn del_cov<B: BoxCoverage + Radius>(&mut self, obj: &B, f: &mut dyn FnMut([usize; 3]))
        -> usize;
}

impl AddCoverage for MapRoot<Coverage> {
    fn add_cov<B: BoxCoverage + Radius>(
        &mut self,
        obj: &B,
        f: &mut dyn FnMut([usize; 3]),
    ) -> usize {
        self.root.node_add_cov(&self.transform, obj, f)
    }
}

impl AddCoverage for MapRoot<DiffCoverage> {
    fn add_cov<B: BoxCoverage + Radius>(
        &mut self,
        obj: &B,
        f: &mut dyn FnMut([usize; 3]),
    ) -> usize {
        self.root.node_add_cov(&self.transform, obj, f)
    }
}

impl DelCoverage for MapRoot<DiffCoverage> {
    fn del_cov<B: BoxCoverage + Radius>(
        &mut self,
        obj: &B,
        f: &mut dyn FnMut([usize; 3]),
    ) -> usize {
        self.root.node_del_cov(&self.transform, obj, f)
    }
}

pub trait NodeAddCoverage {
    fn node_add_cov<B: BoxCoverage + Radius>(
        &mut self,
        transform: &Transform,
        obj: &B,
        f: &mut dyn FnMut([usize; 3]),
    ) -> usize;
}

pub trait NodeDelCoverage {
    fn node_del_cov<B: BoxCoverage + Radius>(
        &mut self,
        transform: &Transform,
        obj: &B,
        f: &mut dyn FnMut([usize; 3]),
    ) -> usize;
}

impl<C: Default + CoverageCheck> MapLike for MapRoot<C> {
    fn new(scale: Vector, origin: Point, width: usize) -> MapRoot<C> {
        Self {
            root: Node::new([0; 3], width),
            transform: Transform { origin, scale },
        }
    }

    fn zero(&mut self) {
        self.root.zero();
    }
}

/// A coverage map which supports adding and subtracting volumes.
pub type DiffCoverageMap = MapRoot<DiffCoverage>;

/// A coverage map which just supports adding volumes.
///
/// If being able to remove volumes is not important, use this
/// data structure as insertion is faster and uses less space.
pub type CoverageMap = MapRoot<Coverage>;

pub struct GridCoverageMap {
    width: usize,
    grid_width: usize,
    transform: Transform,
    counter_map: HashMap<Coord, CubeCounter>,
}

impl GridCoverageMap {
    fn new(scale: Vector, origin: Point, width: usize) -> Self {
        Self {
            width,
            grid_width: 16,
            transform: Transform { origin, scale },
            counter_map: HashMap::new(),
        }
    }

    fn add_cov<B: BoxCoverage + Radius>(
        &mut self,
        sphere: &B,
        f: &mut dyn FnMut([usize; 3]),
    ) -> usize {
        let bb = sphere.bounding_box();

        let lc = self.transform.to_voxel(&bb.lower);
        let uc = self.transform.to_voxel(&bb.upper);

        let lgx = lc[0] / self.grid_width;
        let lgy = lc[1] / self.grid_width;
        let lgz = lc[2] / self.grid_width;

        let ugx = uc[0] / self.grid_width;
        let ugy = uc[1] / self.grid_width;
        let ugz = uc[2] / self.grid_width;

        let mut added = 0;
        for i in lgx..=ugx {
            for j in lgy..=ugy {
                for k in lgz..=ugz {
                    let g = self
                        .counter_map
                        .entry([i, j, k])
                        .or_insert_with(|| CubeCounter::new(self.grid_width));
                    if !g.is_covered() {
                        added += g.add_coverage(
                            &[
                                i * self.grid_width,
                                j * self.grid_width,
                                k * self.grid_width,
                            ],
                            self.grid_width,
                            &self.transform,
                            &sphere.point(),
                            sphere.radius(),
                            f,
                        );
                    }
                }
            }
        }
        added
    }
}

impl MapLike for GridCoverageMap {
    fn new(scale: Vector, origin: Point, width: usize) -> Self {
        GridCoverageMap::new(scale, origin, width)
    }

    fn zero(&mut self) {
        ()
    }
}

impl AddCoverage for GridCoverageMap {
    fn add_cov<B: BoxCoverage + Radius>(
        &mut self,
        obj: &B,
        f: &mut dyn FnMut([usize; 3]),
    ) -> usize {
        GridCoverageMap::add_cov(self, obj, f)
    }
}

/// A naive diff coverage map.  This is just a map
/// of counters for each voxel.  Every time a voxel is covered
/// or uncovered the counters are incremented and decremented.
///
/// This should outperform the other methods when spheres do not
/// overlap significantly.
///
/// Note: This is not a sparse data structure!
#[derive(Clone)]
pub struct NaiveDiffCoverageMap {
    width: usize,
    // The number of actual voxels touched
    voxel_counters: CubeCounter,
    transform: Transform,
}

impl MapLike for NaiveDiffCoverageMap {
    fn new(scale: Vector, origin: Point, width: usize) -> Self {
        Self {
            width,
            voxel_counters: CubeCounter::new(width),
            transform: Transform { origin, scale },
        }
    }

    fn zero(&mut self) {
        self.voxel_counters.zero();
    }
}

impl AddCoverage for NaiveDiffCoverageMap {
    fn add_cov<B: BoxCoverage + Radius>(
        &mut self,
        obj: &B,
        f: &mut dyn FnMut([usize; 3]),
    ) -> usize {
        self.voxel_counters.add_coverage(
            &[0; 3],
            self.width,
            &self.transform,
            &obj.point(),
            obj.radius(),
            f,
        )
    }
}

impl DelCoverage for NaiveDiffCoverageMap {
    fn del_cov<B: BoxCoverage + Radius>(
        &mut self,
        obj: &B,
        f: &mut dyn FnMut([usize; 3]),
    ) -> usize {
        self.voxel_counters.del_coverage(
            &[0; 3],
            self.width,
            &self.transform,
            &obj.point(),
            obj.radius(),
            f,
        )
    }
}
impl Sphere {
    /// Sphere is inside the box
    fn inside_coverage(&self, l: &Point, u: &Point) -> bool {
        self.point[0] - l[0] > self.radius
            && self.point[1] - l[1] > self.radius
            && self.point[2] - l[2] > self.radius
            && u[0] - self.point[0] > self.radius
            && u[1] - self.point[1] > self.radius
            && u[2] - self.point[2] > self.radius
    }

    /// A bounding box is partially covered if any of the sphere is
    /// within it.
    fn partial_coverage(&self, l: &Point, u: &Point) -> bool {
        let mut d = self.radius * self.radius;
        if self.point[0] < l[0] {
            d -= f64::powf(self.point[0] - l[0], 2.0);
        } else if self.point[0] > u[0] {
            d -= f64::powf(self.point[0] - u[0], 2.0);
        }
        if d < 0f64 {
            return false;
        }

        if self.point[1] < l[1] {
            d -= f64::powf(self.point[1] - l[1], 2.0);
        } else if self.point[1] > u[1] {
            d -= f64::powf(self.point[1] - u[1], 2.0);
        }

        if d < 0f64 {
            return false;
        }

        if self.point[2] < l[2] {
            d -= f64::powf(self.point[2] - l[2], 2.0);
        } else if self.point[2] > u[2] {
            d -= f64::powf(self.point[2] - u[2], 2.0);
        }

        d > 0f64
    }

    /// A bounding box is totally covered if the entire volume is
    /// inside the sphere.
    ///
    /// A bounding box is entirely in the sphere when the farthest
    /// corner from the center of the sphere is inside.
    fn total_coverage(&self, l: &Point, u: &Point) -> bool {
        let mut d = self.radius * self.radius;
        let p = self.point;

        d -= f64::max(f64::powf(l[0] - p[0], 2f64), f64::powf(u[0] - p[0], 2f64));
        d -= f64::max(f64::powf(l[1] - p[1], 2f64), f64::powf(u[1] - p[1], 2f64));
        d -= f64::max(f64::powf(l[2] - p[2], 2f64), f64::powf(u[2] - p[2], 2f64));

        d > 0f64
    }
    /// Special case for points.  same idea less computation becuase l == u
    fn point_coverage(&self, l: &Point) -> bool {
        let mut d = self.radius * self.radius;
        let p = self.point;

        d -= f64::powf(l[0] - p[0], 2f64);
        d -= f64::powf(l[1] - p[1], 2f64);
        d -= f64::powf(l[2] - p[2], 2f64);

        d > 0f64
    }
}

pub trait Radius {
    fn radius(&self) -> f64;
    fn point(&self) -> Point;
    fn bounding_box(&self) -> BoundingBox;
}

impl BoxCoverage for Sphere {
    fn box_coverage(&self, bounding_box: &BoundingBox) -> CoverageType {
        let l = bounding_box.lower;
        let u = bounding_box.upper;

        // special case for voxels to speed things up.
        if (l[0] - u[0]).abs() < f64::EPSILON {
            return if self.point_coverage(&l) {
                CoverageType::Total
            } else {
                CoverageType::None
            };
        }

        if self.inside_coverage(&l, &u) {
            CoverageType::Inside
        } else if self.total_coverage(&l, &u) {
            CoverageType::Total
        } else if self.partial_coverage(&l, &u) {
            CoverageType::Partial
        } else {
            CoverageType::None
        }
    }
    /// A bounding box is partially covered if any of the sphere is
    /// within it.
    fn partial_coverage(&self, bounding_box: &BoundingBox) -> bool {
        let l = bounding_box.lower;
        let u = bounding_box.upper;
        let mut d = self.radius * self.radius;
        if self.point[0] < l[0] {
            d -= f64::powf(self.point[0] - l[0], 2.0);
        } else if self.point[0] > u[0] {
            d -= f64::powf(self.point[0] - u[0], 2.0);
        }
        if d < 0f64 {
            return false;
        }

        if self.point[1] < l[1] {
            d -= f64::powf(self.point[1] - l[1], 2.0);
        } else if self.point[1] > u[1] {
            d -= f64::powf(self.point[1] - u[1], 2.0);
        }

        if d < 0f64 {
            return false;
        }

        if self.point[2] < l[2] {
            d -= f64::powf(self.point[2] - l[2], 2.0);
        } else if self.point[2] > u[2] {
            d -= f64::powf(self.point[2] - u[2], 2.0);
        }

        d > 0f64
    }

    /// A bounding box is totally covered if the entire volume is
    /// inside the sphere.
    ///
    /// A bounding box is entirely in the sphere when the farthest
    /// corner from the center of the sphere is inside.
    fn total_coverage(&self, bounding_box: &BoundingBox) -> bool {
        let l = bounding_box.lower;
        let u = bounding_box.upper;
        let mut d = self.radius * self.radius;
        let p = self.point;

        d -= f64::max(f64::powf(l[0] - p[0], 2f64), f64::powf(u[0] - p[0], 2f64));
        d -= f64::max(f64::powf(l[1] - p[1], 2f64), f64::powf(u[1] - p[1], 2f64));
        d -= f64::max(f64::powf(l[2] - p[2], 2f64), f64::powf(u[2] - p[2], 2f64));

        d > 0f64
    }
}

impl Radius for Sphere {
    fn radius(&self) -> f64 {
        self.radius
    }

    fn point(&self) -> Point {
        self.point
    }

    fn bounding_box(&self) -> BoundingBox {
        self.bounding_box()
    }
}

/// Determines how an object covers a bounding box.
///
/// If the volume is a voxel it is either Total, or None.
/// If the volume is a group of voxels it can be Total, Partial or None.
pub trait BoxCoverage {
    fn box_coverage(&self, bounding_box: &BoundingBox) -> CoverageType;
    fn total_coverage(&self, bounding_box: &BoundingBox) -> bool;
    fn partial_coverage(&self, bounding_box: &BoundingBox) -> bool;
}

pub trait CoverageCheck {
    fn is_covered(&self) -> bool;
}

#[derive(Default, Clone)]
pub struct DiffCoverage {
    partial: usize,
    coverage: usize,
}

impl DiffCoverage {
    fn is_dead(&self) -> bool {
        !self.is_partial() && !self.is_covered()
    }
    fn add_part(&mut self) {
        self.partial += 1;
    }

    fn del_part(&mut self) {
        self.partial -= 1;
    }

    fn is_partial(&self) -> bool {
        self.partial > 0
    }

    fn add_cov(&mut self) {
        self.coverage += 1;
    }

    fn del_cov(&mut self) {
        self.coverage -= 1;
    }
}

impl CoverageCheck for DiffCoverage {
    fn is_covered(&self) -> bool {
        self.coverage > 0
    }
}

#[derive(Default, Clone)]
pub struct Coverage {
    covered: bool,
}

impl Coverage {
    fn add_cov(&mut self) {
        self.covered = true;
    }
}

impl CoverageCheck for Coverage {
    fn is_covered(&self) -> bool {
        self.covered
    }
}

impl NodeAddCoverage for Node<DiffCoverage> {
    fn node_add_cov<B: BoxCoverage + Radius>(
        &mut self,
        tf: &Transform,
        obj: &B,
        f: &mut dyn FnMut(Coord),
    ) -> usize {
        let bounding_box = self.bounding_box(tf);
        match obj.box_coverage(&bounding_box) {
            CoverageType::Total => {
                if !self.coverage.is_covered() {
                    self.coverage.add_cov();
                    if let Some(children) = self.children() {
                        let mut change = 0;
                        for child in children.iter_mut() {
                            change += child.node_add_cov(tf, obj, f);
                        }
                        return change;
                    } else if self.width == 1 {
                        f(self.coord);
                        return 1;
                    }
                }
                self.coverage.add_cov();
                0
            }

            CoverageType::Inside | CoverageType::Partial => {
                self.coverage.add_part();
                let children = self
                    .children()
                    .expect("Partially covered nodes should have children");
                let mut change = 0;
                for child in children.iter_mut() {
                    change += child.node_add_cov(tf, obj, f);
                }
                change
            }

            CoverageType::None => 0,
        }
    }
}

impl NodeDelCoverage for Node<DiffCoverage> {
    fn node_del_cov<B: BoxCoverage + Radius>(
        &mut self,
        transform: &Transform,
        obj: &B,
        f: &mut dyn FnMut(Coord),
    ) -> usize {
        let bounding_box = self.bounding_box(transform);
        match obj.box_coverage(&bounding_box) {
            CoverageType::Total => {
                assert!(
                    self.coverage.is_covered(),
                    "Deleted a sphere which was never added"
                );

                self.coverage.del_cov();

                if !self.coverage.is_covered() {
                    let mut change = 0;
                    if let Some(children) = self.get_mut_children() {
                        for child in children.iter_mut() {
                            change += child.node_del_cov(transform, obj, f);
                        }
                        return change;
                    } else if self.width == 1 {
                        f(self.coord);
                        return 1;
                    } else {
                        for i in self.coord[0]..self.coord[0] + self.width {
                            for j in self.coord[1]..self.coord[1] + self.width {
                                for k in self.coord[2]..self.coord[2] + self.width {
                                    f([i, j, k]);
                                }
                            }
                        }
                        return self.width * self.width * self.width;
                    }
                }
            }
            CoverageType::Inside | CoverageType::Partial => {
                self.coverage.del_part();
                if let Some(children) = self.get_mut_children() {
                    let mut change = 0;
                    let mut all_dead = true;
                    for child in children.iter_mut() {
                        change += child.node_del_cov(transform, obj, f);
                        if !child.coverage.is_dead() {
                            all_dead = false;
                        }
                    }

                    if all_dead {
                        self.children = None;
                    }
                    return change;
                }
            }
            CoverageType::None => (),
        }
        0
    }
}

impl Node<Coverage> {
    /// When a node becomes totally covered by an obj,
    /// we need to find all its decendents and subtract their
    /// volume from the total and then remove them.
    ///
    /// In the example below, Layer 2 has a node covered by A, then Layer 3 has a node covered by B,
    /// then layer 1 by C.
    ///
    /// Layer 1 calls compact on its children in layer 2. One node is full thus has no children and
    /// returns 4.  The other node is not full and has children which it calls compact on eventually
    /// returning 2.
    ///
    /// Layer 1        =========== C(8)
    /// Layer 2   A(4) ===== -----
    /// Layer 3              -- == B(2)
    fn fold_children(&self, f: &mut dyn FnMut(Coord)) -> usize {
        if self.coverage.is_covered() {
            self.width * self.width * self.width
        } else {
            let mut voxels_covered = 0;
            if let Some(children) = self.get_children() {
                for child in children.iter() {
                    voxels_covered += child.fold_children(f);
                }
            } else {
                for i in self.coord[0]..self.coord[0] + self.width {
                    for j in self.coord[1]..self.coord[1] + self.width {
                        for k in self.coord[2]..self.coord[2] + self.width {
                            f([i, j, k]);
                        }
                    }
                }
            }
            voxels_covered
        }
    }

    /// If all children are covered, delete them and call this node covered
    fn cleanup(&mut self) {
        if let Some(children) = self.get_mut_children() {
            let mut every_covered = true;
            for child in children.iter_mut() {
                if !child.coverage.is_covered() {
                    every_covered = false;
                    break;
                }
            }

            if every_covered {
                self.children = None;
                self.coverage.add_cov();
            }
        }
    }
}

impl NodeAddCoverage for Node<Coverage> {
    fn node_add_cov<B: BoxCoverage + Radius>(
        &mut self,
        transform: &Transform,
        obj: &B,
        f: &mut dyn FnMut(Coord),
    ) -> usize {
        if self.coverage.is_covered() {
            return 0;
        }
        let bounding_box = self.bounding_box(transform);
        match obj.box_coverage(&bounding_box) {
            CoverageType::Total => {
                let fold = self.fold_children(f);
                let vol = self.width * self.width * self.width;
                self.coverage.add_cov();
                self.children = None;
                vol - fold
            }

            CoverageType::Inside | CoverageType::Partial => {
                let mut change = 0;
                let children = self
                    .children()
                    .expect("Partially covered nodes should have children");
                for child in children.iter_mut() {
                    change += child.node_add_cov(transform, obj, f);
                }
                self.cleanup();
                change
            }
            CoverageType::None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::coverage::{
        AddCoverage, Coverage, CoverageMap, DelCoverage, DiffCoverageMap, MapLike,
        NaiveDiffCoverageMap, Node, Sphere, Transform,
    };
    use approx::relative_eq;
    use std::collections::HashSet;

    #[test]
    fn node_to_bounding_box() {
        // A node (octant) which has a width of 4, (4x4x4 voxels).
        let node: Node<Coverage> = Node::new([0; 3], 4);

        // If we say that the node has the following transformation to
        // real coordinates
        let transform = Transform {
            origin: [0f64; 3],
            scale: [2.2; 3],
        };

        // Then we would expect a bounding box from the center of the lowest
        // voxel (0,0,0) which is (1.1, 1.1, 1.1) to the center of the upper
        // voxel which is at (3,3,3) and would have a center at (7.7, 7.7, 7.7)
        let aabb = node.bounding_box(&transform);
        relative_eq!(aabb.lower[0], 1.1);
        relative_eq!(aabb.lower[1], 1.1);
        relative_eq!(aabb.lower[2], 1.1);

        relative_eq!(aabb.upper[0], 7.7);
        relative_eq!(aabb.upper[1], 7.7);
        relative_eq!(aabb.upper[2], 7.7);
    }

    #[test]
    fn naive_diff_map() {
        let mut nmt = NaiveDiffCoverageMap::new([1.0120482f64; 3], [-9.0, 11.0, -18.0], 256);
        assert_eq!(
            nmt.add_cov(
                &Sphere::new([35.319, 51.331, 18.794], 4.675f64),
                &mut |_| ()
            ),
            412
        );
        assert_eq!(
            nmt.add_cov(
                &Sphere::new([34.624, 50.910, 20.029], 4.675f64),
                &mut |_| ()
            ),
            93
        );
    }

    // The numbers used here come from examples run in TEMPy.
    #[test]
    fn diff_map() {
        let mut mt = DiffCoverageMap::new([1.0120482f64; 3], [-9.0, 11.0, -18.0], 256);
        assert_eq!(
            mt.add_cov(
                &Sphere::new([35.319, 51.331, 18.794], 4.675f64),
                &mut |_| ()
            ),
            412
        );

        assert_eq!(
            mt.del_cov(
                &Sphere::new([35.319, 51.331, 18.794], 4.675f64),
                &mut |_| ()
            ),
            412
        );

        assert_eq!(
            mt.add_cov(
                &Sphere::new([34.624, 50.910, 20.029], 4.675f64),
                &mut |_| ()
            ),
            418
        );
        assert_eq!(
            mt.del_cov(
                &Sphere::new([34.624, 50.910, 20.029], 4.675f64),
                &mut |_| ()
            ),
            418
        );

        assert_eq!(
            mt.add_cov(
                &Sphere::new([35.227, 49.610, 20.593], 4.675f64),
                &mut |_| ()
            ),
            406
        );
        assert_eq!(
            mt.del_cov(
                &Sphere::new([35.227, 49.610, 20.593], 4.675f64),
                &mut |_| ()
            ),
            406
        );

        assert_eq!(
            mt.add_cov(
                &Sphere::new([35.319, 51.331, 18.794], 4.675f64),
                &mut |_| ()
            ),
            412
        );
        assert_eq!(
            mt.add_cov(
                &Sphere::new([34.624, 50.910, 20.029], 4.675f64),
                &mut |_| ()
            ),
            93
        );
        assert_eq!(
            mt.del_cov(
                &Sphere::new([35.319, 51.331, 18.794], 4.675f64),
                &mut |_| ()
            ),
            87
        );
        assert_eq!(
            mt.del_cov(
                &Sphere::new([34.624, 50.910, 20.029], 4.675f64),
                &mut |_| ()
            ),
            418
        );
    }

    #[test]
    fn mono_map() {
        let mut mt = CoverageMap::new([1.0120482f64; 3], [-9.0, 11.0, -18.0], 256);
        assert_eq!(
            mt.add_cov(
                &Sphere::new([35.319, 51.331, 18.794], 4.675f64),
                &mut |_| ()
            ),
            412
        );
        assert_eq!(
            mt.add_cov(
                &Sphere::new([34.624, 50.910, 20.029], 4.675f64),
                &mut |_| ()
            ),
            93
        );
        assert_eq!(
            mt.add_cov(
                &Sphere::new([35.227, 49.610, 20.593], 4.675f64),
                &mut |_| ()
            ),
            87
        );
        mt.add_cov(
            &Sphere::new([35.227, 49.610, 20.593], 4.675f64),
            &mut |_| (),
        );
    }

    #[test]
    fn same_output_map() {
        let mut dif = DiffCoverageMap::new([1.0120482f64; 3], [-9.0, 11.0, -18.0], 256);
        let mut cov = CoverageMap::new([1.0120482f64; 3], [-9.0, 11.0, -18.0], 256);
        let mut dif_out = HashSet::new();
        let mut cov_out = HashSet::new();
        assert_eq!(
            dif.add_cov(
                &Sphere::new([35.319, 51.331, 18.794], 4.675f64),
                &mut |coord| {
                    dif_out.insert(coord);
                }
            ),
            412
        );

        assert_eq!(
            dif.add_cov(
                &Sphere::new([34.624, 50.910, 20.029], 4.675f64),
                &mut |coord| {
                    dif_out.insert(coord);
                }
            ),
            93
        );
        assert_eq!(
            cov.add_cov(
                &Sphere::new([35.319, 51.331, 18.794], 4.675f64),
                &mut |coord| {
                    cov_out.insert(coord);
                }
            ),
            412
        );

        assert_eq!(
            cov.add_cov(
                &Sphere::new([34.624, 50.910, 20.029], 4.675f64),
                &mut |coord| {
                    cov_out.insert(coord);
                }
            ),
            93
        );
        assert_eq!(cov_out.len(), dif_out.len());
    }
}
