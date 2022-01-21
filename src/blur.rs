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

//! Functions for blurring structures and building simulated density maps.

use crate::geom::{Coord, Distance, Point, Transform, Vector};

/// A 3D gaussian like function with a width of sigma.
pub struct Gaussian {
    center: Point,
    exp_denom: f64,
}

impl Gaussian {
    /// Create a new 3D gaussian with width of sigma.
    ///
    /// # Arguments
    ///
    /// * `center` - The center of the gaussian.
    /// * `sigma` - An array of sigma for each dimension.
    pub fn new(center: Point, sigma: f64) -> Self {
        Gaussian {
            center,
            exp_denom: -1f64 / (2f64 * sigma * sigma),
        }
    }

    /// TODO, better function name.
    pub fn height_at_point(&self, point: &Point) -> f64 {
        let v = self.center.vector_to(point);
        let w = (v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]);
        f64::exp(w * self.exp_denom)
    }
}

/// Similiar to TEMPy Gaussian blur but atoms are not snapped to 1A grid.
///
/// We assume that an atom can me modelled as a 3D gaussian
/// and that the density at each voxel can be obtained by integregating
/// the atoms density over the volume of the voxel.
///
/// The density drops off rapidly for voxels far from the atomic
/// center so we provide a cutoff parameter: the distance from the center at which to
/// stop calculating density.
pub struct GaussianBlur {
    cutoff: f64,
}

impl GaussianBlur {
    pub fn new(cutoff: f64) -> Self {
        Self { cutoff }
    }
}

/// Add and remove atoms from simulated maps by using a blurrer.
///
/// A blurrer should implement this trait to allow molecules to
/// be added and removed from a density map.
pub trait AddGaussianToMap {
    /// Add an atom to the map by blurring it and applying density to
    /// the corresponding voxels.
    ///
    /// # Arguments
    ///
    /// * `map` - the density map to update.
    /// * `position` - the atom to blur onto the map.
    /// * `height` - the height of the gaussian.
    fn add_gaussian(&self, map: &mut Map, position: &Point, height: f64, sigma: f64);
}

impl AddGaussianToMap for GaussianBlur {
    fn add_gaussian(&self, map: &mut Map, position: &Point, height: f64, sigma: f64) {
        let gaus = Gaussian::new(*position, sigma);
        let cutoff_dist = self.cutoff * sigma;

        // Define a box to blur in.
        let pos = position;
        let lower_corner = [
            pos[0] - cutoff_dist,
            pos[1] - cutoff_dist,
            pos[2] - cutoff_dist,
        ];

        let upper_corner = [
            pos[0] + cutoff_dist,
            pos[1] + cutoff_dist,
            pos[2] + cutoff_dist,
        ];

        let tf = &map.transform;
        let mut lv = tf.to_voxel(&lower_corner);
        let mut uv = tf.to_voxel(&upper_corner);

        uv[0] = usize::min(map.size[0] - 1, uv[0]);
        uv[1] = usize::min(map.size[1] - 1, uv[1]);
        uv[2] = usize::min(map.size[2] - 1, uv[2]);

        let dens = &mut map.densities;
        // We should blur voxels with in radius of cutoff but for now just do
        // every voxel in the box.
        for (row, k) in dens[lv[2]..=uv[2]].iter_mut().zip(lv[2]..=uv[2]) {
            for (col, j) in row[lv[1]..=uv[1]].iter_mut().zip(lv[1]..=uv[1]) {
                for (cell, i) in col[lv[0]..=uv[0]].iter_mut().zip(lv[0]..=uv[0]) {
                    let center = tf.to_real(&[i, j, k]);
                    *cell += gaus.height_at_point(&center) * height;
                }
            }
        }
    }
}

pub struct Map {
    size: [usize; 3],
    pub transform: Transform,
    pub densities: Vec<Vec<Vec<f64>>>,
}

pub trait GetDensity {
    /// Get the density value at the coordinate
    fn get_density(&self, coord: Coord) -> f64;
}

impl Map {
    /// Create a new Map
    ///
    /// # Example
    /// ```
    /// // Create a map with origin (0,0,0) cubic cell width of 1A, and 100^3 voxels
    /// use voxcov::blur::Map;
    /// let map = Map::new([0f64; 3], [1.0f64; 3], [100; 3]);
    /// ```
    pub fn new(origin: Point, scale: Vector, size: [usize; 3]) -> Self {
        Self {
            size,
            transform: Transform { origin, scale },
            densities: vec![vec![vec![0.0; size[0]]; size[1]]; size[2]],
        }
    }

    pub fn zero(&mut self) {
        for row in self.densities.iter_mut() {
            for col in row.iter_mut() {
                for cell in col.iter_mut() {
                    *cell = 0f64;
                }
            }
        }
    }
}

impl GetDensity for Map {
    fn get_density(&self, coord: Coord) -> f64 {
        self.densities[coord[2]][coord[1]][coord[0]]
    }
}

pub struct RawMap {
    pub transform: Transform,
    pub densities: Vec<f64>,
    size: [usize; 3],
}

impl RawMap {
    pub fn new(origin: Point, scale: Vector, size: [usize; 3], vec: Vec<f64>) -> Self {
        Self {
            size,
            transform: Transform { origin, scale },
            densities: vec,
        }
    }
}

impl GetDensity for RawMap {
    fn get_density(&self, coord: Coord) -> f64 {
        self.densities[coord[0] + coord[1] * self.size[2] + coord[2] * self.size[1] * self.size[2]]
    }
}
