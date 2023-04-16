use std::time::Instant;
use voxcov::blur::{AddGaussianToMap, GaussianBlur, Map, MultiGaussianBlur};

/// Stolen from https://github.com/ybyygu/rust-octree/blob/master/examples/demo.rs
fn read_points_xyz(txt: &str) -> Vec<[f64; 3]> {
    let mut positions = Vec::new();
    for line in txt.lines() {
        let attrs: Vec<_> = line.split_whitespace().collect();
        let (_symbol, position) = attrs.split_first().expect("empty line");
        assert_eq!(position.len(), 3, "{:?}", position);
        let p: Vec<f64> = position.iter().map(|x| x.parse().unwrap()).collect();
        positions.push([p[0], p[1], p[2]]);
    }

    positions
}

fn main() {
    let stream = include_str!("data/5fj8.xyz");
    let points = read_points_xyz(stream);
    println!("Points: {}", points.len());

    let mut map = Map::new([0.0; 3], [1.084; 3], [256; 3]);
    let blurrer = GaussianBlur::new(4.0);
    let now = Instant::now();
    for p in &points {
        blurrer.add_gaussian(&mut map, p, 1.0, 9.0 * 0.187);
    }
    println!("End blur {:?}", now.elapsed());

    let now = Instant::now();
    for p in &points {
        blurrer.add_gaussian(&mut map, p, -1.0, 9.0 * 0.187);
        blurrer.add_gaussian(&mut map, p, 1.0, 9.0 * 0.187);
    }
    println!("End blur {:?}", now.elapsed());
    let now = Instant::now();
    map.zero();
    println!("End zero {:?}", now.elapsed());
    let now = Instant::now();
    let mut mgb = MultiGaussianBlur::new(9.0);
    for p in &points {
       mgb.track(*p);
    }
    mgb.apply(&mut map);
    println!("End zero {:?}", now.elapsed());
}
