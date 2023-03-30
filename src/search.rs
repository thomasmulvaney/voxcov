use crate::geom::{Coord, Point, Transform};
use std::collections::hash_map::HashMap;

pub struct DamageTracker {
    cells: HashMap<Coord, usize>,
    transform: Transform,
}

impl DamageTracker {
    pub fn new(cell_size: f64) -> Self {
        Self {
            cells: HashMap::new(),
            transform: Transform::new([0f64; 3], [cell_size; 3]),
        }
    }

    pub fn reset(&mut self) {
        for val in self.cells.values_mut() {
            *val = 0;
        }
    }

    /// Add a point to track
    pub fn track(&mut self, point: Point) {
        let coord = self.transform.to_voxel(&point);
        self.cells.entry(coord).and_modify(|x| *x += 1).or_insert(1);
    }

    /// Check if point has had neighbours which updated.
    pub fn check(&mut self, point: Point) -> bool {
        let coord = self.transform.to_voxel(&point);
        for i in 0..=2 {
            for j in 0..=2 {
                for k in 0..=2 {
                    let mut c = coord;
                    c[0] += i - 1;
                    c[1] += j - 1;
                    c[2] += k - 1;
                    if let Some(v) = self.cells.get(&c) {
                        if *v > 0 {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
}
