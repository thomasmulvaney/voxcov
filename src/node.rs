use crate::geom::{BoundingBox, Coord, Distance, Point, Transform};
use crate::grid::{Grid, GridLike};

/// A node in the tree.
///
/// Every node corresponds to a cubic group of one or more voxels.
/// The coordinates are the lowest coordinates of the cube.
///
/// The node is generic over `C` an object which holds information about coverage.
#[derive(Clone)]
pub struct Node<C> {
    pub coord: Coord,
    pub width: usize,
    pub coverage: C,
    pub children: Option<Box<[Node<C>; 8]>>,
}

/// Octree implementation over a generic Node type which can have different
/// coverage implementations.
impl<C: Default> Node<C> {
    pub fn new(coord: Coord, width: usize) -> Node<C> {
        Node {
            coord,
            width,
            coverage: C::default(),
            children: None,
        }
    }

    pub fn zero(&mut self) {
        self.children = None;
        self.coverage = C::default();
    }

    pub fn is_voxel(&self) -> bool {
        self.width == 1
    }

    pub fn bounding_box(&self, transform: &Transform) -> BoundingBox {
        let mut lower = transform.to_real(&self.coord);

        lower[0] += transform.scale[0] / 2f64;
        lower[1] += transform.scale[1] / 2f64;
        lower[2] += transform.scale[2] / 2f64;

        let mut upper_coord = self.coord;
        upper_coord[0] += self.width;
        upper_coord[1] += self.width;
        upper_coord[2] += self.width;

        let mut upper = transform.to_real(&upper_coord);
        upper[0] -= transform.scale[0] / 2f64;
        upper[1] -= transform.scale[1] / 2f64;
        upper[2] -= transform.scale[2] / 2f64;

        BoundingBox { lower, upper }
    }

    pub fn get_mut_children(&mut self) -> Option<&mut Box<[Node<C>; 8]>> {
        self.children.as_mut()
    }

    pub fn get_children(&self) -> &Option<Box<[Node<C>; 8]>> {
        &self.children
    }

    pub fn children(&mut self) -> Option<&mut Box<[Node<C>; 8]>> {
        if self.is_voxel() {
            return None;
        } else if self.children.is_none() {
            let d = self.width / 2;
            let x = self.coord[0];
            let y = self.coord[1];
            let z = self.coord[2];
            self.children = Some(Box::new([
                // 0: lx, ly, lz
                Node::new([x, y, z], d),
                // 1: ux, ly, lz
                Node::new([x + d, y, z], d),
                // 2: lx, uy, lz
                Node::new([x, y + d, z], d),
                // 3: lx, ly, uz
                Node::new([x, y, z + d], d),
                // 4: ux, uy, lz
                Node::new([x + d, y + d, z], d),
                // 5: ux, ly, uz
                Node::new([x + d, y, z + d], d),
                // 6: lx, lu, uz
                Node::new([x, y + d, z + d], d),
                // 7: ux, lu, uz
                Node::new([x + d, y + d, z + d], d),
            ]));
        }
        self.children.as_mut()
    }
}
/// Dense datastructure for counting voxel coverage
#[derive(Clone)]
pub struct CubeCounter {
    counters: Grid<u16>,
}

impl CubeCounter {
    pub fn new(width: usize) -> Self {
        Self {
            counters: Grid::new([width; 3]),
        }
    }
    pub fn add_coverage(
        &mut self,
        offset: &Coord,
        width: usize,
        tf: &Transform,
        pos: &Point,
        cutoff_dist: f64,
        f: &mut dyn FnMut([usize; 3]),
    ) -> usize {
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

        let lv = tf.to_voxel(&lower_corner);
        let uv = tf.to_voxel(&upper_corner);

        let xl = if lv[0] <= offset[0] {
            0
        } else {
            lv[0] - offset[0]
        };
        let yl = if lv[1] <= offset[1] {
            0
        } else {
            lv[1] - offset[1]
        };
        let zl = if lv[2] <= offset[2] {
            0
        } else {
            lv[2] - offset[2]
        };

        let xu = if uv[0] < offset[0] {
            width
        } else {
            usize::min(width, uv[0] - offset[0])
        };

        let yu = if uv[1] < offset[1] {
            width
        } else {
            usize::min(width, uv[1] - offset[1])
        };

        let zu = if uv[2] < offset[2] {
            width
        } else {
            usize::min(width, uv[2] - offset[2])
        };

        let cut2 = cutoff_dist * cutoff_dist;
        let mut added = 0;
        self.counters.rec([xl, yl, zl]);
        self.counters.rec([xu, yu, zu]);
        for k in zl..zu {
            for j in yl..yu {
                for i in xl..xu {
                    // Lowest point of voxel
                    let coord: Coord = [i, j, k];
                    let mut point: Point = tf.to_real(&coord);

                    // Center of voxel
                    point[0] += tf.scale[0] / 2f64;
                    point[1] += tf.scale[1] / 2f64;
                    point[2] += tf.scale[2] / 2f64;

                    let dist2 = point.dist2(pos);

                    // is center of voxel within radius?
                    if dist2 < cut2 {
                        // Count newly covered voxel
                        if let Some(val) = self.counters.get_mut(coord) {
                            if *val == 0 {
                                added += 1;
                                f(coord);
                            }
                            *val += 1;
                        }
                    }
                }
            }
        }
        added
    }

    pub fn del_coverage(
        &mut self,
        offset: &Coord,
        width: usize,
        tf: &Transform,
        pos: &Point,
        cutoff_dist: f64,
        f: &mut dyn FnMut([usize; 3]),
    ) -> usize {
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

        let lv = tf.to_voxel(&lower_corner);
        let uv = tf.to_voxel(&upper_corner);

        let xl = if lv[0] <= offset[0] {
            0
        } else {
            lv[0] - offset[0]
        };
        let yl = if lv[1] <= offset[1] {
            0
        } else {
            lv[1] - offset[1]
        };
        let zl = if lv[2] <= offset[2] {
            0
        } else {
            lv[2] - offset[2]
        };

        let xu = if uv[0] < offset[0] {
            width
        } else {
            usize::min(width, uv[0] - offset[0])
        };

        let yu = if uv[1] < offset[1] {
            width
        } else {
            usize::min(width, uv[1] - offset[1])
        };

        let zu = if uv[2] < offset[2] {
            width
        } else {
            usize::min(width, uv[2] - offset[2])
        };

        let cut2 = cutoff_dist * cutoff_dist;
        let mut deleted = 0;
        for k in zl..zu {
            for j in yl..yu {
                for i in xl..xu {
                    // Lowest point of voxel
                    let coord: Coord = [i, j, k];
                    let mut point: Point = tf.to_real(&coord);

                    // Center of voxel
                    point[0] += tf.scale[0] / 2f64;
                    point[1] += tf.scale[1] / 2f64;
                    point[2] += tf.scale[2] / 2f64;

                    let dist2 = point.dist2(pos);

                    // is center of voxel within radius?
                    if dist2 < cut2 {
                        // Count newly covered voxel
                        if let Some(val) = self.counters.get_mut(coord) {
                            if *val == 1 {
                                deleted += 1;
                                f(coord);
                            }
                            *val -= 1;
                        }
                    }
                }
            }
        }
        deleted
    }

    pub fn zero(&mut self) {
        self.counters.zero();
    }
}
