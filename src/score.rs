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

pub struct MI {
    matrix: Vec<Vec<usize>>,
    xs: Vec<usize>,
    ys: Vec<usize>,
    x_low: f64,
    y_low: f64,
    x_step: f64,
    y_step: f64,
    x_bins: usize,
    y_bins: usize,
    tot: usize,
    thresh: f64,
}

impl MI {
    fn new(x_low: f64, x_high: f64, y_low: f64, y_high: f64, x_bins: usize, y_bins: usize) -> MI {
        let res = MI {
            x_low,
            y_low,
            x_bins,
            y_bins,
            x_step: (x_high - x_low) / (x_bins as f64),
            y_step: (y_high - y_low) / (y_bins as f64),
            matrix: vec![vec![0; y_bins + 1]; x_bins + 1],
            xs: vec![0; x_bins + 1],
            ys: vec![0; y_bins + 1],
            tot: 0,
            thresh: 0.0,
        };
        //println!("step: {} {}", res.x_step, res.y_step);
        res
    }

    fn x_bin(&self, val: f64) -> usize {
        //println!("{} {} {}", val, self.x_bins, self.x_low);
        let r = ((val - self.x_low) / self.x_step as f64) as usize;
        //println!("{}", r);
        r
    }

    fn y_bin(&self, val: f64) -> usize {
        ((val - self.y_low) / self.y_step as f64) as usize
    }

    fn add_val(&mut self, s: f64, e: f64) {
        let sb = self.x_bin(s);
        let eb = self.y_bin(e);
        self.matrix[sb][eb] += 1;
        self.xs[sb] += 1;
        self.ys[eb] += 1;
        self.tot += 1;
    }

    fn del_val(&mut self, s: f64, e: f64) {
        let sb = self.x_bin(s);
        let eb = self.y_bin(e);
        self.matrix[sb][eb] -= 1;
        self.xs[sb] -= 1;
        self.ys[eb] -= 1;
        self.tot -= 1;
    }

    fn get_val(&self) -> f64 {
        let mut hx = 0.0;
        let mut hy = 0.0;
        let mut hxy = 0.0;
        for i in 0..self.x_bins {
            let x = self.xs[i] as f64 / self.tot as f64;
            if x > self.thresh {
                hx += (-x * x.log(2.0));
            }
        }
        for j in 0..self.y_bins {
            let y = self.ys[j] as f64 / self.tot as f64;
            if y > self.thresh {
                hy += (-y * y.log(2.0))
            }
        }

        for i in 0..self.x_bins {
            for j in 0..self.y_bins {
                let xy = self.matrix[i][j] as f64 / self.tot as f64;
                if xy > self.thresh {
                    hxy += (-xy * xy.log(2.0))
                }
            }
        }
        //println!("{} {} {}", hx, hy, hxy);
        hx + hy - hxy
        //hy - hxy
    }

    fn zero(&mut self) {
        for i in 0..self.x_bins {
            self.xs[i] = 0;
        }

        for j in 0..self.y_bins {
            self.ys[j] = 0;
        }

        for i in 0..self.x_bins {
            for j in 0..self.y_bins {
                self.matrix[i][j] = 0;
            }
        }
    }
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
            if let Some(s) = sim.get_density(coord) {
                if let Some(e) = exp.get_density(coord) {
                    moc.add_val(*s, *e)
                }
            }
            if sim.get_density(coord).is_none() != exp.get_density(coord).is_none() {
                println!(
                    "Oops! {:?} {:?} {:?}",
                    coord,
                    sim.get_size(),
                    exp.get_size()
                );
            }
            // TODO: We might actually want to track cases where map and model
            // have pixels which don't exist...
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
            if let Some(s) = sim.get_density(coord) {
                if let Some(e) = exp.get_density(coord) {
                    moc.del_val(*s, *e)
                }
            }
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

pub struct SMI<M> {
    map: M,
    moc: MI,
}

impl<M> SMI<M>
where
    M: MapLike + AddCoverage + DelCoverage,
{
    /// Create a new SMOC scoring object.
    ///
    /// To initialize requires the dimensions of the maps to be used.
    pub fn new(
        scale: Vector,
        origin: Point,
        width: usize,
        x_low: f64,
        x_high: f64,
        y_low: f64,
        y_high: f64,
        x_bins: usize,
        y_bins: usize,
    ) -> Self {
        Self {
            map: M::new(scale, origin, width),
            moc: MI::new(x_low, x_high, y_low, y_high, x_bins, y_bins),
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
            if let Some(s) = sim.get_density(coord) {
                if let Some(e) = exp.get_density(coord) {
                    moc.add_val(*s, *e)
                }
            }
            if sim.get_density(coord).is_none() != exp.get_density(coord).is_none() {
                println!(
                    "Oops! {:?} {:?} {:?}",
                    coord,
                    sim.get_size(),
                    exp.get_size()
                );
            }
            // TODO: We might actually want to track cases where map and model
            // have pixels which don't exist...
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
            if let Some(s) = sim.get_density(coord) {
                if let Some(e) = exp.get_density(coord) {
                    moc.del_val(*s, *e)
                }
            }
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
