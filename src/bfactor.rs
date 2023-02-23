use crate::blur::{AddGaussianToMap, Gaussian, Gaussian1D, GaussianBlur, GetDensity, Map, RawMap};
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
    cutoff_b: f64,

    // Central intensity of noise
    noise: f64,
    noise_sigma: f64,
}

/// Compute the center of mass (weighted mean) of a set of points.
struct MassCenter {
    mx: f64,
    my: f64,
    mz: f64,
    tot: f64,
}

impl MassCenter {
    /// Initialize for no points.
    fn new() -> Self {
        Self {
            mx: 0f64,
            my: 0f64,
            mz: 0f64,
            tot: 0f64,
        }
    }

    /// Add a point and its mass to the set of points.
    fn add_mass(&mut self, point: &Point, mass: f64) {
        self.mx += mass * point[0];
        self.my += mass * point[1];
        self.mz += mass * point[2];
        self.tot += mass;
    }

    /// Compute the center of mass of the points.
    fn center(&self) -> Point {
        [self.mx / self.tot, self.my / self.tot, self.mz / self.tot] as Point
    }
}

struct Moment {
    center: Point,
    dx2: f64,
    dy2: f64,
    dz2: f64,
    mass: f64,
}

impl Moment {
    fn new(center: Point) -> Moment {
        Moment {
            center,
            dx2: 0f64,
            dy2: 0f64,
            dz2: 0f64,
            mass: 0f64,
        }
    }

    fn add_mass(&mut self, point: &Point, mass: f64) {
        self.dx2 += (self.center[0] - point[0]).powf(2.0) * mass;
        self.dy2 += (self.center[1] - point[1]).powf(2.0) * mass;
        self.dz2 += (self.center[2] - point[2]).powf(2.0) * mass;
        self.mass += mass;
    }

    fn variance(&self) -> f64 {
        let px = (self.dx2 / self.mass).sqrt();
        let py = (self.dy2 / self.mass).sqrt();
        let pz = (self.dz2 / self.mass).sqrt();
        (px + py + pz) / 3.0
    }
}

impl BfactorEstimator {
    pub fn new(cutoff: f64, exp_map: RawMap, sim_map: Map) -> Self {
        let noise = 5.36e-6 * 1f64;
        let noise_sigma = exp_map.std() * 1000000.0;
        println!("Noise: {:.2e} {:.2e}", noise, noise_sigma);
        let mut t = exp_map;
        let mut s = sim_map;
        Self {
            t,
            s,
            cutoff: 2.5,
            cutoff_b: 2.5,
            noise,
            noise_sigma,
        }
    }

    // Zero out the simulated map
    pub fn zero(&mut self) {
        self.s.zero()
    }

    // Add a gaussian to the simulated map
    pub fn add_gaussian(&mut self, pos: &Point, height: f64, sigma: f64) {
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
        // We should blur voxels with in radius of cutoff but for now just do
        // every voxel in the box.
        dens.rec(uv);
        dens.rec(lv);
        for i in lv[0]..=uv[0] {
            for j in lv[1]..=uv[1] {
                for k in lv[2]..=uv[2] {
                    let center = tf.to_real(&[i, j, k]);
                    let v = gaus.pdf(&center) * height;
                    let t = *self.t.get_density([i, j, k]).unwrap();
                    assert!(!v.is_nan(), "C: {:?} P: {:?}", center, pos);
                    if v < 0.0 {
                        println!("oop! {}", v);
                    }
                    if let Some(d) = dens.get_mut([i, j, k]) {
                        *d += v;
                    }
                }
            }
        }
    }

    // Normalize the simulated map
    pub fn normalize(&mut self) {
        let mut tot = 0f64;
        let mut cnt = 0f64;
        let mut var = 0f64;
        let mut len = 0f64;
        for i in 0..self.t.size[0] {
            for j in 0..self.t.size[1] {
                for k in 0..self.t.size[2] {
                    let t = *self.t.get_density([i, j, k]).unwrap();
                    let s = *self.s.get_density([i, j, k]).unwrap();
                    if s > 0.0 {
                        let nom = if t < 2e-6 { 1.0 } else { 0.0 };
                        let denom = s + nom;
                        let w = nom / denom;
                        cnt += w;
                        tot += w;
                        len += 1f64;
                    }
                }
            }
        }
        let mean = tot / cnt;
        println!(
            "{:.3e} mean: {:.3e}, pi: {:.3e} len: {:.3e}",
            self.noise,
            mean,
            tot / len,
            len,
        );
        let mut var_tot = 0f64;
        for i in 0..self.t.size[0] {
            for j in 0..self.t.size[1] {
                for k in 0..self.t.size[2] {
                    let t = *self.t.get_density([i, j, k]).unwrap();
                    let s = *self.s.get_density([i, j, k]).unwrap();
                    if s > 0.0 {
                        let nom = 1e-6; //noise.pdf(t);
                        let denom = s + nom;
                        let w = nom / denom;
                        var += w * ((mean - t) * (mean - t));
                        var_tot += w;
                    }
                }
            }
        }
        let bg_sig = (var / var_tot).sqrt();
        let bg_sig_d = (bg_sig - self.noise_sigma);
        println!(
            "BG: u:{:.2e} n:{:.2e} s:{:.2e} {:.2e}",
            mean, self.noise_sigma, bg_sig, bg_sig_d
        );
        self.noise_sigma = self.noise_sigma + bg_sig_d;
        println!("D::: {:.3e}", mean - self.noise);
        //self.noise = tot / cnt;
    }

    pub fn likelihood(&self) -> f64 {
        let mut tot_s = 0f64;
        let mut tot = 0f64;
        for i in 0..self.t.size[0] {
            for j in 0..self.t.size[1] {
                for k in 0..self.t.size[2] {
                    let s = *self.s.get_density([i, j, k]).unwrap();
                    tot_s += s;
                }
            }
        }
        for i in 0..self.t.size[0] {
            for j in 0..self.t.size[1] {
                for k in 0..self.t.size[2] {
                    let t = *self.t.get_density([i, j, k]).unwrap();
                    let s = *self.s.get_density([i, j, k]).unwrap();
                    tot += s * t;
                }
            }
        }
        tot
    }

    // Compute bfactor for atom
    pub fn bfactor(&mut self, pos: &Point, height: f64, sigma: f64) -> f64 {
        let gaus = Gaussian::new(*pos, sigma);
        let cutoff_dist = self.cutoff_b * sigma;

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

        let mut moment = Moment::new(*pos);
        for i in lv[0]..=uv[0] {
            for j in lv[1]..=uv[1] {
                for k in lv[2]..=uv[2] {
                    let center: Point = tf.to_real(&[i, j, k]);
                    let v = gaus.pdf(&center) * height;
                    let t = *self.t.get_density([i, j, k]).unwrap_or(&0f64);
                    let s = *self.s.get_density([i, j, k]).unwrap_or(&0f64);
                    let nom = v;
                    let noise_d = 5.72e-6;
                    let denom = s;
                    let gamma = nom / denom;
                    assert!(
                        denom >= nom,
                        "denom {} < nom {} t{} v{} s{} sig{}",
                        denom,
                        nom,
                        t,
                        v,
                        s,
                        sigma
                    );
                    let w = gamma * t;
                    moment.add_mass(&center, w);
                }
            }
        }
        let vg = moment.variance();
        let vg = f64::max(vg, 0.1);
        vg
    }

    /// Compute the new mean.  This is the center of mass of the responsibility
    /// of the gaussian multiplied by the experimental density.
    pub fn mean(&mut self, pos: &Point, height: f64, sigma: f64) -> Point {
        let gaus = Gaussian::new(*pos, sigma);
        let cutoff_dist = self.cutoff_b * sigma;

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

        let mut com = MassCenter::new();
        for i in lv[0]..=uv[0] {
            for j in lv[1]..=uv[1] {
                for k in lv[2]..=uv[2] {
                    let center: Point = tf.to_real(&[i, j, k]);
                    assert!(center[0] >= 0f64);
                    assert!(center[1] >= 0f64);
                    assert!(center[2] >= 0f64);
                    let v = gaus.pdf(&center) * height;
                    let t = *self.t.get_density([i, j, k]).unwrap_or(&0f64);
                    let s = *self.s.get_density([i, j, k]).unwrap_or(&0f64);
                    let nom = v;
                    let denom = s;
                    // Gamma aka the responsibility.  If the denominator, is 0,
                    // set the responsibility to zero.  Gamma should always be
                    // between 0 and 1.
                    let gamma = if denom <= 0f64 { 0f64 } else { nom / denom };
                    assert!(
                        gamma >= 0.0,
                        "g: {}, d: { }, n: {}, s: {}, p: {:?}",
                        gamma,
                        denom,
                        nom,
                        s,
                        [i, j, k]
                    );
                    let w = gamma * t;
                    assert!(t >= 0f64);
                    assert!(w >= 0f64);
                    com.add_mass(&center, w);
                }
            }
        }
        com.center()
    }
}
