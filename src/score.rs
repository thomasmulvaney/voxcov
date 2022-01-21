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

//! Various scoring functions
//!
use crate::blur::GetDensity;
use crate::coverage::{AddCoverage, BoxCoverage, DelCoverage, MapLike, Radius};
use crate::geom::{Coord, Point, Vector};

/// # A Manders Overlap Coefficient
///
/// This consists of three sums SS, SE and EE.
/// TODO: Describe the three sums.
pub struct MOC {
    ss: f64,
    ee: f64,
    se: f64,
}

impl MOC {
    /// Create a new Manders Overlap Coefficient with sums set to zero.
    fn new() -> Self {
        MOC {
            ss: 0f64,
            ee: 0f64,
            se: 0f64,
        }
    }

    /// Add two pairs of values to the score
    fn add_val(&mut self, s: f64, e: f64) {
        self.ss += s * s;
        self.ee += e * e;
        self.se += s * e;
    }

    /// Remove two pairs of values to the score
    fn del_val(&mut self, s: f64, e: f64) {
        self.ss -= s * s;
        self.ee -= e * e;
        self.se -= s * e;
    }

    /// Get the current value of the score
    fn get_val(&self) -> f64 {
        self.se / f64::sqrt(self.ss * self.ee)
    }

    // Reset the sums to zero
    fn zero(&mut self) {
        self.ss = 0f64;
        self.se = 0f64;
        self.ee = 0f64;
    }
}

/// # SMOC - Segmented Manders Overlap Coefficient
///
/// A SMOC score calculated with the help of a coverage map.
///
/// The API is centred around adding and removing atoms from the coverage map
/// thus forming a 'segment' of covered voxels.
///
/// A typical 'segment' might be a residue or window-of-residues.  In
/// this case, sets of atoms are added or removed from the score using
/// the `add_cov` and `del_cov` methods.  The score of the set is then
/// obtained by calling `get_val`.
pub struct SMOC<M> {
    map: M,
    moc: MOC,
}

impl<M> SMOC<M>
where
    M: MapLike + AddCoverage + DelCoverage,
{
    /// Create a new SMOC scoring object.
    ///
    /// To initialize requires the dimensions of the maps to be used.
    pub fn new(scale: Vector, origin: Point, width: usize) -> Self {
        Self {
            map: M::new(scale, origin, width),
            moc: MOC::new(),
        }
    }

    /// Given a sphere and two maps update the SMOC score with the value of any
    /// voxels in the sphere which have not been considered.
    ///
    /// # Arguments
    /// - `obj` a sphere
    /// - `sim` a simulated map
    /// - `exp` an experimental map
    pub fn add_cov<B: BoxCoverage + Radius, D: GetDensity, C: GetDensity>(
        &mut self,
        obj: &B,
        sim: &C,
        exp: &D,
    ) {
        let map = &mut self.map;
        let moc = &mut self.moc;
        map.add_cov(obj, &mut |coord: Coord| {
            moc.add_val(sim.get_density(coord), exp.get_density(coord))
        });
    }

    /// Given a sphere and two maps update the SMOC score by removing the value of any
    /// voxels in the sphere which are no longer to be considered.
    ///
    /// # Arguments
    /// - `obj` a sphere
    /// - `sim` a simulated map
    /// - `exp` an experimental map
    pub fn del_cov<B: BoxCoverage + Radius, D: GetDensity, C: GetDensity>(
        &mut self,
        obj: &B,
        sim: &D,
        exp: &C,
    ) {
        let map = &mut self.map;
        let moc = &mut self.moc;
        map.del_cov(obj, &mut |coord: Coord| {
            moc.del_val(sim.get_density(coord), exp.get_density(coord))
        });
    }

    /// Get the MOC value of the current segment.
    pub fn get_val(&self) -> f64 {
        self.moc.get_val()
    }

    /// Reset the MOC accumulator and uncover all voxels.
    pub fn zero(&mut self) {
        self.moc.zero();
        self.map.zero();
    }
}

#[cfg(test)]
mod test {
    use super::MOC;
    #[test]
    fn moc_test() {
        let mut moc = MOC::new();
        moc.add_val(1f64, 1f64);
        assert_eq!(moc.get_val(), 1f64);
    }
}
