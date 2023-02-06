use crate::blur::{AddGaussianToMap, Gaussian, GaussianBlur, GetDensity, Map, RawMap};
use crate::geom::{Distance, Point};
use crate::grid::{Grid, GridLike};

pub struct BfactorEstimator {
    // The experimental map
    t: RawMap,
    // The simulated map
    s: Map,
    // Cutoff for gaussians, don't go bigger than 4.0.
    // Bigger is slower and at 4.0 stdev its pretty much zero.
    cutoff: f64,

    noise: f64,
}

impl BfactorEstimator {
    pub fn new(cutoff: f64, exp_map: RawMap, sim_map: Map) -> Self {
        let mut t = exp_map;
        let mut s = sim_map;
        let noise = 0.000000000001;
        Self {
            t,
            s,
            cutoff,
            noise,
        }
    }

    // Zero out the simulated map
    pub fn zero(&mut self) {
        self.s.zero()
    }

    // Add a gaussian to the simulated map
    pub fn add_gaussian(&mut self, pos: &Point, height: f64, sigma: f64) {
        let blur = GaussianBlur::new(4f64);
        blur.add_gaussian(&mut self.s, pos, height, sigma);
    }

    // Normalize the simulated map
    pub fn normalize(&mut self) {
        //let min_v: f64 = *self.s.densities.grid.iter().min_by(|a, b| a.partial_cmp(b).expect(&format!("oh no {} {}", a, b))).unwrap();
        let mut tot = 0f64;
        let min_v = 0f64;

        for i in 0..self.s.densities.grid.len() {
            tot += self.s.densities.grid[i];
        }

        for i in 0..self.s.densities.grid.len() {
            self.s.densities.grid[i] = (self.s.densities.grid[i] - min_v) / tot;
        }
    }

    // Compute bfactor for atom
    pub fn bfactor(&mut self, pos: &Point, height: f64, sigma: f64) -> f64 {
        let mut vx = 0f64;
        let mut vy = 0f64;
        let mut vz = 0f64;
        let mut tot = 0f64;
        let gaus = Gaussian::new(*pos, sigma);
        let cutoff_dist = self.cutoff * sigma;

        // Define a box to blur in.
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

        let tf = &self.t.transform;
        let lv = tf.to_voxel(&lower_corner);
        let mut uv = tf.to_voxel(&upper_corner);

        uv[0] = usize::min(self.t.size[0] - 1, uv[0]);
        uv[1] = usize::min(self.t.size[1] - 1, uv[1]);
        uv[2] = usize::min(self.t.size[2] - 1, uv[2]);

        let dens = &mut self.s.densities;

        // We track the boxes that we have modified.
        // This makes zeroing faster as we don't have zero the entire map,
        // just the parts which have changed.
        dens.rec(uv); // upper corner
        dens.rec(lv); // lower corner

        for i in lv[0]..=uv[0] {
            for j in lv[1]..=uv[1] {
                for k in lv[2]..=uv[2] {
                    let center: Point = tf.to_real(&[i, j, k]);
                    let v = gaus.height_at_point(&center) * height;
                    let t = self.t.get_density([i, j, k]).unwrap();
                    let s = self.s.get_density([i, j, k]).unwrap();
                    let dx = pos[0] - center[0]; // center.dist2(pos);
                    let dy = pos[1] - center[1]; // center.dist2(pos);
                    let dz = pos[2] - center[2]; // center.dist2(pos);
                    let dx2 = dx * dx;
                    let dy2 = dy * dy;
                    let dz2 = dz * dz;
                    let prod = t * v;
                    let norm = s * t;
                    let norm = norm + self.noise;

                    let post = if norm == 0f64 { 0f64 } else { prod / norm };
                    let w = t * post;
                    vx += dx2 * w;
                    vy += dy2 * w;
                    vz += dz2 * w;
                    tot += w;
                }
            }
        }
        let sx = vx / tot;
        let sy = vy / tot;
        let sz = vz / tot;
        //println!("{} {} {}", sx,sy,sz);
        (sx + sy + sz) / 3.0
    }
}
