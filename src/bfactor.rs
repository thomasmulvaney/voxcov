use crate::blur::{AddGaussianToMap, Gaussian, Gaussian1D, GaussianBlur, GetDensity, Map, RawMap};
use crate::geom::{Coord, Distance, Point};
use crate::grid::{Grid, GridLike};
use crate::score::LinearBin;
use crate::search::DamageTracker;
use std::time::{Duration, Instant};

pub struct BoundingBoxIter {
    low: Coord,
    high: Coord,
    c: Coord,
    done: bool,
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
        Self {
            low,
            high,
            c: low,
            done: false,
        }
    }
}

impl Iterator for BoundingBoxIter {
    type Item = Coord;
    fn next(&mut self) -> Option<Coord> {
        if self.done {
            return None;
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
        return Some(ret);
    }
}

trait EM {
    fn sample(&self, data: &mut Map);
    fn maximise(&mut self, data: &Map, obs: &Map, back: f64, sigma: f64);
    fn mean(&self, data: &Map, obs: &Map) -> Point;
    fn likelihood(&self, obs: &Map) -> f64;
}

struct GaussianComp {
    mean: Point,
    sigma: f64,
    prev_sigma: f64,
    prev_mean: Point,
    weight: f64,
    changed: bool,
}

struct UniformComp {
    mean: f64,
    prev_mean: f64,
    // Measuring the variance is useful for thresholding.
    sigma: f64,
    changed: bool,
}

impl GaussianComp {
    fn new(mean: Point, sigma: f64, weight: f64) -> Self {
        Self {
            mean,
            sigma,
            prev_sigma: 0.0,
            prev_mean: [0.0; 3],
            weight,
	    changed: true,
        }
    }

    fn set_mean(&mut self, mean: Point) {
	if mean.dist2(&self.mean) > 0.01 {
	    self.prev_mean = self.mean;
            self.mean = mean;
	    self.changed = true;
	} else {
	    self.changed = false;
	}
    }

    fn set_sigma(&mut self, sigma: f64) {
	if (self.prev_sigma - sigma).abs() > 0.1 {
	    self.prev_sigma = self.sigma;
	    self.sigma = sigma;
	    self.changed = true;
	} else {
	    self.changed = false;
	}
    }

    fn has_changed(&self) -> bool {
	self.changed
    }
}

impl UniformComp {
    fn new(mean: f64) -> Self {
        Self {
            mean,
            sigma: 0.0,
            prev_mean: 0.0,
	    changed: true,
        }
    }

    fn set_mean(&mut self, mean: f64) {
        if (self.mean - mean).abs() > 1e-7 {
	    self.prev_mean = self.mean;
	    self.mean = mean;
	    self.changed = true
	} else {
	    self.changed = true
	}
	    
    }

    fn has_changed(&self) -> bool {
	self.changed
    }
}

struct Mean {
    tot: f64,
    count: usize,
}

impl Mean {
    fn new() -> Self {
        Self {
            tot: 0f64,
            count: 0,
        }
    }

    fn add(&mut self, v: f64) {
        self.tot += v;
        self.count += 1;
    }

    fn val(&self) -> f64 {
        self.tot / self.count as f64
    }
}

struct Variance {
    mean: f64,
    tot: f64,
    count: usize,
}

impl Variance {
    fn new(mean: f64) -> Self {
        Self {
            mean,
            tot: 0f64,
            count: 0,
        }
    }

    fn add(&mut self, v: f64) {
        self.tot += (self.mean - v).powf(2.0);
        self.count += 1;
    }

    fn val(&self) -> f64 {
        (self.tot / self.count as f64).sqrt()
    }
}

impl EM for GaussianComp {
    fn sample(&self, data: &mut Map) {
        let blur = GaussianBlur::new(4.0);
        if self.has_changed() {
            if self.prev_sigma > 0.0 {
                blur.add_gaussian(data, &self.prev_mean, self.weight * -1.0, self.prev_sigma);
            }
            blur.add_gaussian(data, &self.mean, self.weight, self.sigma);
        }
    }

    fn maximise(&mut self, data: &Map, obs: &Map, back: f64, sigma: f64) {
        let pos = self.mean;
        let gaus = Gaussian::new(self.mean, self.sigma);
        let cutoff_dist = self.sigma * 4.0;
        let tf = &obs.transform;
        let bb = data.get_bounding_coord(pos, cutoff_dist);

        let mut bi = BoundingBoxIter::new(bb.lower, bb.upper);
        let mut moment = Moment::new(self.mean);

        for coord in bi {
            let center = tf.to_real(&coord);
            if self.mean.dist(&center) <= cutoff_dist {
                let nom = gaus.pdf(&center) * self.weight;
                let t = *obs.get_density(coord).unwrap_or(&0f64);
                let denom = *data.get_density(coord).unwrap_or(&0f64);
                let gamma = nom / denom;
		let resp = gamma * t;
		moment.add_mass(&center, resp);
            }
        }
        self.sigma = moment.variance().unwrap_or(5.0);
    }

    fn mean(&self, data: &Map, obs: &Map) -> Point {
        let pos = self.mean;
        let gaus = Gaussian::new(self.mean, self.sigma);
        let cutoff_dist = f64::max(0.5, self.sigma) * 4.0;
        let tf = &obs.transform;
        let bb = data.get_bounding_coord(pos, cutoff_dist);

        let mut bi = BoundingBoxIter::new(bb.lower, bb.upper);
        let mut com = MassCenter::new();

        for coord in bi {
            let center = tf.to_real(&coord);
            if self.mean.dist(&center) <= cutoff_dist {
                let v = gaus.pdf(&center) * self.weight;
                let t = *obs.get_density(coord).unwrap_or(&0f64);
                let denom = *data.get_density(coord).unwrap_or(&0f64);
                let nom = v;
                let gamma = nom / denom;
		let resp = gamma * t;
		com.add_mass(&center, resp);
            }
        }
	// If the new center of mass is not defined, just return current mean.
        com.center().unwrap_or(self.mean)
    }

    fn likelihood(&self, obs: &Map) -> f64 {
        0f64
    }
}

impl EM for UniformComp {
    fn sample(&self, data: &mut Map) {
        if self.has_changed() {
            data.add(self.mean - self.prev_mean);
        }
    }
    fn maximise(&mut self, data: &Map, obs: &Map, back: f64, sigma: f64) {
        self.prev_mean = self.mean;
        let dens = &data.densities;
        let tot = data.sum();
        let mut bi = BoundingBoxIter::new(dens.low.unwrap(), dens.high.unwrap());
        let mut mean = Mean::new();

        for (t, d) in obs.densities.grid.iter().zip(data.densities.grid.iter()) {
            let denom = d / tot;
            let gamma = self.mean / denom;
            let resp = gamma * t;
            if *d > 0.0 {
                mean.add(resp);
            }
        }

        let mut var = Variance::new(self.mean);

        for (t, d) in obs.densities.grid.iter().zip(data.densities.grid.iter()) {
            let denom = d / tot;
            let gamma = self.mean / denom;
            let resp = gamma * t;
            if *d > 0.0 {
                var.add(resp);
            }
        }
        self.sigma = var.val();
        let old_mean = self.mean;
        let new_mean = mean.val();
        let mean_delta = new_mean - self.mean;
        let boosted = self.mean + (mean_delta * 100.0);
        if boosted > self.mean * 0.01 && boosted < self.mean * 100.0 {
            self.mean = boosted;
        } else {
            self.mean = new_mean;
        }
    }

    fn mean(&self, data: &Map, obs: &Map) -> Point {
        [0f64; 3]
    }
    fn likelihood(&self, obs: &Map) -> f64 {
        0f64
    }
}

pub struct Mixture {
    gaussians: Vec<GaussianComp>,
    noise: UniformComp,
    obs: Map,
    pub data: Map,
    dt: DamageTracker,
}

impl Mixture {
    pub fn moc(&self) -> f64 {0 }
    pub fn new(data: Map, obs: RawMap) -> Self {
        println!("Init mean: {}", obs.mean());
        let mut ob = data.clone();
        let size = data.size;

        for i in 0..size[0] {
            for j in 0..size[1] {
                for k in 0..size[2] {
                    let t = obs.get_density([i, j, k]).unwrap();
                    if let Some(d) = ob.densities.get_mut([i, j, k]) {
                        *d += t;
                    }
                }
            }
        }
        Self {
            gaussians: vec![],
            noise: UniformComp::new(obs.mean() * 1.0),
            data,
            obs: ob,
            dt: DamageTracker::new(16.0),
        }
    }

    pub fn add_gaussian(&mut self, mean: Point, sigma: f64, weight: f64) {
        self.gaussians.push(GaussianComp::new(mean, sigma, weight));
    }

    /// Set the mean of a gaussian manually.
    pub fn set_mean(&mut self, idx: usize, mean: Point) {
        if let Some(g) = self.gaussians.get_mut(idx) {
            g.set_mean(mean);
        } else {
            println!("No gaussian with index: {}", idx);
        }
    }

    pub fn step_internal(&mut self) -> usize {
        let updates = self.expect();
        self.maximise();
        updates
    }

    pub fn step(&mut self) {
        self.data.zero();
        self.noise.prev_mean = 0.0;
        for g in &mut self.gaussians {
            g.prev_sigma = 0.0;
        }
        for i in 0..10 {
            if self.step_internal() == 0 {
                return ();
            }
        }
    }

    /// Run an expectation cycle.  Sample from the distribution,
    /// adding to data.
    fn expect(&mut self) -> usize {
        let mut updates = 0;
        self.dt.reset();
        for g in &self.gaussians {
            if g.has_changed() || self.noise.has_changed() {
                updates += 1;
                self.dt.track(g.mean);
            }
            g.sample(&mut self.data);
        }
        println!("Updates: {} {}", updates, self.gaussians.len());
        self.noise.sample(&mut self.data);
        updates
    }

    pub fn mean(&self, idx: usize) -> Point {
        self.gaussians[idx].mean(&self.data, &self.obs)
    }

    pub fn noise(&self) -> f64 {
        self.noise.mean
    }

    pub fn set_noise(&mut self, noise: f64) {
        self.noise.mean = noise;
    }

    pub fn sigma(&self, idx: usize) -> f64 {
        self.gaussians[idx].sigma
    }

    pub fn set_sigma(&mut self, idx: usize, sigma: f64) {
	self.gaussians[idx].set_sigma(sigma);
    }

    fn maximise(&mut self) {
        for g in &mut self.gaussians {
            if g.has_changed() || self.dt.check(g.mean) || self.noise.has_changed() {
                g.maximise(&self.data, &self.obs, self.noise.mean, self.noise.sigma);
            }
        }
        self.noise.maximise(&self.data, &self.obs, 0.0, 0.0);
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
    fn center(&self) -> Option<Point> {
	if self.tot > 0f64 {
            return Some([self.mx / self.tot, self.my / self.tot, self.mz / self.tot])
	}
	None
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

    fn variance(&self) -> Option<f64> {
	if self.mass > 0f64 {
        let px = (self.dx2 / self.mass).sqrt();
        let py = (self.dy2 / self.mass).sqrt();
        let pz = (self.dz2 / self.mass).sqrt();
        return Some((px + py + pz) / 3.0)
	}
	None
    }
}
