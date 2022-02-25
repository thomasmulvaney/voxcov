use crate::geom::Coord;
#[derive(Clone)]
pub struct Grid<T> {
    grid: Vec<T>,
    size: Coord,
    low: Option<Coord>,
    high: Option<Coord>,
}

impl<T> Grid<T> {
    pub fn new(size: Coord) -> Self
    where
        T: Default + Copy,
    {
        Self {
            size,
            grid: vec![T::default(); size[0] * size[1] * size[2]],
            low: None,
            high: None,
        }
    }
}

pub trait GridLike<T> {
    /// Get the value T at coordinate.
    fn get(&self, coord: Coord) -> Option<&T>;

    /// Record a bounding coordinate.
    /// This updates the bounding box.
    fn rec(&mut self, coord: Coord);

    /// Get a mutable reference to the value at the coordinate.
    fn get_mut(&mut self, coord: Coord) -> Option<&mut T>
    where
        T: Default + Copy;

    /// Set the value at coordinate..
    fn set(&mut self, coord: Coord, val: T);

    /// Reset all values to there default value
    fn zero(&mut self)
    where
        T: Default + Copy;
}

impl<T> GridLike<T> for Grid<T>
where
    T: Copy,
{
    fn set(&mut self, coord: Coord, val: T) {
        self.grid[coord[0] + coord[1] * self.size[2] + coord[2] * self.size[1] * self.size[2]] =
            val;

        // track damaged region with a bounding box
        self.rec(coord);
    }

    fn get(&self, coord: Coord) -> Option<&T> {
        self.grid
            .get(coord[0] + coord[1] * self.size[2] + coord[2] * self.size[1] * self.size[2])
    }

    fn rec(&mut self, coord: Coord) {
        if let Some(ref mut low) = self.low {
            low[0] = usize::min(coord[0], low[0]);
            low[1] = usize::min(coord[1], low[1]);
            low[2] = usize::min(coord[2], low[2]);
        } else {
            self.low = Some(coord);
        }

        if let Some(ref mut high) = self.high {
            high[0] = usize::max(coord[0], high[0]);
            high[1] = usize::max(coord[1], high[1]);
            high[2] = usize::max(coord[2], high[2]);
        } else {
            self.high = Some(coord);
        }
    }

    fn get_mut(&mut self, coord: Coord) -> Option<&mut T>
    where
        T: Default + Copy,
    {
        self.grid
            .get_mut(coord[0] + coord[1] * self.size[2] + coord[2] * self.size[1] * self.size[2])
    }

    fn zero(&mut self)
    where
        T: Default + Copy,
    {
        // iterate over any region which has been touched.
        if let Some(low) = self.low {
            if let Some(high) = self.high {
                for i in low[0]..=high[0] {
                    for j in low[1]..=high[1] {
                        for k in low[2]..=high[2] {
                            self.set([i, j, k], T::default());
                        }
                    }
                }
            }
        }
        self.low = None;
        self.high = None;
    }
}
