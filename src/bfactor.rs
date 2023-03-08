use crate::blur::{AddGaussianToMap, Gaussian, Gaussian1D, GaussianBlur, GetDensity, Map, RawMap};
use crate::geom::{Coord, Distance, Point};
use crate::grid::{Grid, GridLike};
use crate::score::LinearBin;

pub struct BoundingBoxIter {
    low: Coord,
    high: Coord,
    c: Coord,
    done: bool
}

impl BoundingBoxIter {
    /// # Example
    /// ```
    /// // Create an iterator which visits all coordinates in a bounding box.
    /// use voxcov::bfactor::BoundingBoxIter;
    /// let mut bi = BoundingBoxIter::new([0; 3], [2; 3]);
    /// assert!(bi.next() == Some([0, 0, 0]));
    /// assert!(bi.next() == Some([0, 0, 1]));
    /// assert!(bi.next() == Some([0, 0, 2]));
    /// assert!(bi.next() == Some([0, 1, 0]));
    /// assert!(bi.next() == Some([0, 1, 1]));
    /// assert!(bi.next() == Some([0, 1, 2]));
    /// assert!(bi.next() == Some([0, 2, 0]));
    /// assert!(bi.next() == Some([0, 2, 1]));
    /// assert!(bi.next() == Some([0, 2, 2]));
    /// assert!(bi.next() == Some([1, 0, 0]));
    /// assert!(bi.next() == Some([1, 0, 1]));
    /// assert!(bi.next() == Some([1, 0, 2]));
    /// assert!(bi.next() == Some([1, 1, 0]));
    /// assert!(bi.next() == Some([1, 1, 1]));
    /// assert!(bi.next() == Some([1, 1, 2]));
    /// assert!(bi.next() == Some([1, 2, 0]));
    /// assert!(bi.next() == Some([1, 2, 1]));
    /// assert!(bi.next() == Some([1, 2, 2]));
    /// assert!(bi.next() == Some([2, 0, 0]));
    /// assert!(bi.next() == Some([2, 0, 1]));
    /// assert!(bi.next() == Some([2, 0, 2]));
    /// assert!(bi.next() == Some([2, 1, 0]));
    /// assert!(bi.next() == Some([2, 1, 1]));
    /// assert!(bi.next() == Some([2, 1, 2]));
    /// assert!(bi.next() == Some([2, 2, 0]));
    /// assert!(bi.next() == Some([2, 2, 1]));
    /// assert!(bi.next() == Some([2, 2, 2]));
    /// assert!(bi.next() == None);
    /// ```
    pub fn new(low: Coord, high: Coord) -> Self {
	Self {low, high, c: low, done: false}
    }
}

impl Iterator for BoundingBoxIter {
    type Item = Coord;
    fn next(&mut self) -> Option<Coord> {
	if self.done {
	    return None
	}
	let ret = self.c;
	if self.c[2] < self.high[2] {
            self.c[2] += 1;
	    return Some(ret);
	} else {
            self.c[2] = self.low[2];
        }
	
	if self.c[1] < self.high[1] {
            self.c[1] += 1;
	    return Some(ret);
	} else {
            self.c[1] = self.low[1];
        }

	if self.c[0] < self.high[0] {
            self.c[0] += 1;
	    return Some(ret);
	} else {
	    self.done = true;
	}
	return Some(ret)
    }
}

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
    dd: DiscreteDist,
    ud: UniformDist,
    gd: GaussianDist,
}

trait EM {
    fn sample(&self, data: &mut Map, obs: &RawMap);
    fn maximise(&mut self, data: &Map, obs: &RawMap);
    fn likelihood(&self, obs: &RawMap) -> f64;
}

struct GaussianComp {
    mean: Point,
    sigma: f64,
    weight: f64,
}

impl GaussianComp {
    fn new(mean: Point, sigma: f64, weight: f64) -> Self {
        Self {
            mean,
            sigma,
            weight,
        }
    }

    fn set_mean(&mut self, mean: Point) {
        self.mean = mean;
    }
}

impl EM for GaussianComp {
    fn sample(&self, data: &mut Map, obs: &RawMap) {
        let pos = self.mean;
        let gaus = Gaussian::new(self.mean, self.sigma);
        let cutoff_dist = self.sigma * 3.0;
        let tf = &obs.transform;
	let bb = data.get_bounding_coord(pos, cutoff_dist);
        let dens = &mut data.densities;
        dens.rec(bb.lower);
        dens.rec(bb.upper);
	let mut bi = BoundingBoxIter::new(bb.lower, bb.upper);
	for coord in bi {
	    let center = tf.to_real(&coord);
	    let v = gaus.pdf(&center) * self.weight;
	    assert!(!v.is_nan(), "C: {:?} P: {:?}", center, pos);
	    if let Some(d) = dens.get_mut(coord) {
		*d += v;
	    }
	}
    }
    fn maximise(&mut self, data: &Map, obs: &RawMap) {
	let pos = self.mean;
        let gaus = Gaussian::new(self.mean, self.sigma);
        let cutoff_dist = self.sigma * 3.0;
        let tf = &obs.transform;
	let bb = data.get_bounding_coord(pos, cutoff_dist);
	let mut com = MassCenter::new();
	let mut bi = BoundingBoxIter::new(bb.lower, bb.upper);
	for coord in bi {
	    let center = tf.to_real(&coord);
	    let v = gaus.pdf(&center) * self.weight;
	    let t = *obs.get_density(coord).unwrap_or(&0f64);
	    let denom = *data.get_density(coord).unwrap_or(&0f64);
	    let nom = v;
	    let gamma = nom / denom;
	    let resp = gamma * t;
	    com.add_mass(&center, resp);
	}
	self.set_mean(com.center())
    }
    fn likelihood(&self, obs: &RawMap) -> f64 {
        0f64
    }
}

struct Mixture {
    gaussians: Vec<GaussianComp>,
    noise: UniformDist,
    obs: RawMap,
    data: Map,
}

impl Mixture {
    fn new(data: Map, obs: RawMap) -> Self {
        Self {
            gaussians: vec![],
            noise: UniformDist::new(),
            data,
            obs,
        }
    }

    fn add_gaussian(&mut self, mean: Point, sigma: f64, weight: f64) {
        self.gaussians.push(GaussianComp::new(mean, sigma, weight));
    }

    /// Set the mean of a gaussian manually.
    fn set_mean(&mut self, idx: usize, mean: Point) {
        if let Some(g) = self.gaussians.get_mut(idx) {
            g.set_mean(mean);
        } else {
            println!("No gaussian with index: {}", idx);
        }
    }

    /// Run an expectation cycle.
    fn expect(&mut self, data: &mut Map, obs: &RawMap) {
        self.data.zero();
        for g in &self.gaussians {
            g.sample(&mut self.data, &self.obs);
        }
        self.noise.sample(&mut self.data, &self.obs);
    }

    fn likelihood(&self) -> f64 {
        0f64
    }
    fn gaussian_likelihood(&self, idx: usize) -> f64 {
        0f64
    }
    fn noise_likelihood(&self) -> f64 {
        0f64
    }
}

/// Store the probabilities as a discreet probability density function.
struct DiscreteDist {
    lb: LinearBin,
    probs: Vec<usize>,
    total: usize,
    peak: usize,
}

struct GaussianDist {
    mean: f64,
    weight: f64,
    sigma: f64,
}

struct UniformDist {
    mean: f64,
    weight: f64,
}

impl UniformDist {
    fn new() -> Self {
        UniformDist {
            mean: 1.0,
            weight: 1.0,
        }
    }

    fn maximise(&mut self, t: &RawMap, s: &Map) {
        let mut tot = 0f64;
        let mut resp_tot = 0f64;
        let mut cnt = 0f64;
        let lo = s.densities.low.unwrap();
        let hi = s.densities.high.unwrap();
        for i in lo[0]..hi[0] {
            for j in lo[1]..hi[1] {
                for k in lo[2]..hi[2] {
                    let t = *t.get_density([i, j, k]).unwrap();
                    let s = *s.get_density([i, j, k]).unwrap();
                    if s > 0.0 {
                        let nom = self.prob();
                        let denom = s + nom;
                        let gamma = nom / denom;
                        let resp = gamma * t;
                        cnt += 1f64;
                        tot += gamma;
                        resp_tot += resp;
                    }
                }
            }
        }
        let weight = tot / cnt;
        let mean = resp_tot / cnt;
        println!(
            "M: {} -> {}, W: {} -> {} >>> {}",
            self.mean, mean, self.weight, weight, resp_tot
        );
        self.mean = mean;
        self.weight = weight;
    }

    fn prob(&self) -> f64 {
        self.mean // * self.weight
    }
}

impl EM for UniformDist {
    fn sample(&self, data: &mut Map, obs: &RawMap) {
        let lo = data.densities.low.unwrap();
        let hi = data.densities.high.unwrap();
	let mut bi = BoundingBoxIter::new(lo,hi);
	for coord in bi {
	    if let Some(d) = data.densities.get_mut(coord) {
		*d += self.mean;
	    }
        }
    }
    fn maximise(&mut self, data: &Map, obs: &RawMap) {}

    fn likelihood(&self, obs: &RawMap) -> f64 {
        0f64
    }
}

impl GaussianDist {
    fn new() -> Self {
        Self {
            mean: 1.0,
            weight: 1.0,
            sigma: 1.0,
        }
    }

    fn maximise(&mut self, t: &RawMap, s: &Map) {
        let mut tot = 0f64;
        let mut resp_tot = 0f64;
        let mut cnt = 0f64;
        let lo = s.densities.low.unwrap();
        let hi = s.densities.high.unwrap();
        for i in lo[0]..hi[0] {
            for j in lo[1]..hi[1] {
                for k in lo[2]..hi[2] {
                    let t = *t.get_density([i, j, k]).unwrap();
                    let s = *s.get_density([i, j, k]).unwrap();
                    if s >= 0.0 {
                        let nom = self.prob(t);
                        let denom = s + nom;
                        let gamma = nom / denom;
                        let resp = gamma * t;
                        cnt += t; //1f64;
                        tot += gamma;
                        resp_tot += resp;
                    }
                }
            }
        }

        let weight = resp_tot / cnt;
        let mean = resp_tot / tot;
        let mut var = 0f64;
        let mut gamma_tot = 0f64;
        for i in lo[0]..hi[0] {
            for j in lo[1]..hi[1] {
                for k in lo[2]..hi[2] {
                    let t = *t.get_density([i, j, k]).unwrap();
                    let s = *s.get_density([i, j, k]).unwrap();
                    if s >= 0.0 {
                        let nom = self.prob(t);
                        let denom = s + nom;
                        let gamma = nom / denom;
                        let delta = t - mean;
                        let delta2 = delta * delta;
                        gamma_tot += gamma * t;
                        var += delta2 * gamma * t;
                    }
                }
            }
        }

        let sigma = (var / gamma_tot).powf(0.5);
        println!(
            "M: {} -> {}, W: {} -> {}, V: {} ",
            self.mean, mean, self.weight, weight, sigma
        );
        self.mean = mean;
        self.weight = weight;
        self.sigma = sigma;
    }

    fn prob(&self, v: f64) -> f64 {
        let g = Gaussian1D::new(self.mean, self.sigma);
        g.pdf(v) * self.weight
    }

    fn val(&self) -> f64 {
        self.weight * self.mean
    }
}

impl DiscreteDist {
    fn new(lo: f64, hi: f64, bins: usize) -> Self {
        Self {
            lb: LinearBin::new(lo, hi, bins),
            probs: vec![0; bins],
            total: 0,
            peak: 0,
        }
    }

    fn add(&mut self, v: f64) {
        let bin = self.lb.bin(v).unwrap();
        self.probs[bin] += 1;
        self.total += 1;
    }

    fn prob(&self, v: f64) -> f64 {
        let bin = self.lb.bin(v).unwrap_or(0);
        self.probs[bin] as f64 / self.total as f64
    }

    fn percentile(&self, v: f64) -> f64 {
        let mut tot = 0;
        let mut idx = 0;
        for p in &self.probs {
            tot += p;
            let perc = tot as f64 / self.total as f64;
            if perc > v {
                return self.lb.low + (idx as f64 * self.lb.step);
            }
            idx += 1;
        }
        return 0.0;
    }

    fn peak_idx(&mut self) -> usize {
        if self.peak > 0 {
            return self.peak;
        }
        let mut prev = 0;
        let mut idx = 0;
        let mut peak_idx = 0;
        for p in &self.probs {
            if *p > prev {
                prev = *p;
                peak_idx = idx;
            }
            idx += 1;
        }
        println!("PPPP: {}", peak_idx);
        self.peak = peak_idx;
        peak_idx
    }

    fn prob_new(&mut self, v: f64) -> f64 {
        let bin = self.lb.bin(v).unwrap_or(0);
        let peak_bin = self.peak_idx();
        if bin < peak_bin {
            return self.probs[bin] as f64 / self.total as f64;
        } else if bin < 2 * peak_bin {
            let delta = bin - peak_bin;
            let low = self.probs[peak_bin - delta] as f64 / self.total as f64;
            return low;
        }
        return 0.0;
    }

    fn info(&self) {
        println!(
            "Lo: {} Hi: {} Step: {}",
            self.lb.low, self.lb.high, self.lb.step
        );
        for p in &self.probs[300..600] {
            //println!("P: {}", *p as f64 / self.total as f64);
        }
    }
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
        //println!("Noise: {:.2e} {:.2e}", noise, noise_sigma);
        let min = exp_map.min();
        let max = exp_map.max();
        let mut dd = DiscreteDist::new(min, max, 100000);

        for v in &exp_map.densities {
            dd.add(*v);
        }
        dd.info();

        let mut t = exp_map;
        let mut s = sim_map;
        Self {
            t,
            s,
            dd,
            cutoff: 3.0,
            cutoff_b: 3.0,
            noise,
            noise_sigma,
            ud: UniformDist::new(),
            gd: GaussianDist::new(),
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
                    assert!(!v.is_nan(), "C: {:?} P: {:?}", center, pos);
                    if let Some(d) = dens.get_mut([i, j, k]) {
                        *d += v;
                    }
                }
            }
        }
    }

    // Normalize the simulated map
    pub fn normalize(&mut self) {
        self.ud.maximise(&self.t, &self.s);
        self.gd.maximise(&self.t, &self.s);
        return ();
        let mut tot = 0f64;
        let mut cnt = 0f64;
        let lo = self.s.densities.low.unwrap();
        let hi = self.s.densities.high.unwrap();
        for i in lo[0]..hi[0] {
            for j in lo[1]..hi[1] {
                for k in lo[2]..hi[2] {
                    let t = *self.t.get_density([i, j, k]).unwrap();
                    let noise = self.dd.prob_new(t); // * 0.25;
                    let s = *self.s.get_density([i, j, k]).unwrap();
                    if s > 0.0 {
                        let nom = noise * self.noise;
                        let denom = s + nom;
                        let gamma = nom / denom;
                        let resp = gamma * t;
                        cnt += 1f64;
                        tot += gamma;
                    }
                }
            }
        }
        let weight = tot / cnt;
        println!("weight: {:.3e} -> {:.3e}", self.noise, weight);
        self.noise = weight;
    }

    pub fn likelihood(&self) -> f64 {
        let mut tot = 0f64;
        for i in 0..self.t.size[0] {
            for j in 0..self.t.size[1] {
                for k in 0..self.t.size[2] {
                    let t = *self.t.get_density([i, j, k]).unwrap();
                    let s = *self.s.get_density([i, j, k]).unwrap();
                    if s > 0f64 {
                        tot += f64::ln(s * t);
                    } //
                }
            }
        }
        tot
    }

    // Compute bfactor for atom
    pub fn bfactor(&mut self, pos: &Point, height: f64, sigma: f64) -> f64 {
        let gaus = Gaussian::new(*pos, sigma);
        let cutoff_dist = self.cutoff_b * sigma;

        //let noise = self.dd.percentile(0.75);
        //println!("NOISE A: {}", noise);

        //let noise = self.dd.percentile(0.9999999999);
        //println!("NOISE B: {}", noise);
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
                    let noise = 0.0; //self.gd.prob(t);
                    let nom = v;
                    let denom = s + noise; // + self.dd.prob(v); //(self.noise * 5.0);
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

        let mut com  = MassCenter::new();
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
                    let denom = s + 1e-3;
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
