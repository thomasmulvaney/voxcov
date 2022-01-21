use std::time::Instant;
use voxcov::coverage::{
    AddCoverage, Coverage, DelCoverage, DiffCoverage, MapLike, MapRoot, NaiveDiffCoverageMap,
};
use voxcov::geom::Sphere;

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

    let mut hmap = NaiveDiffCoverageMap::new([1.0f64; 3], [0f64; 3], 1024);
    let mut map = MapRoot::<Coverage>::new([1.0f64; 3], [0f64; 3], 1024);
    let mut dmap = MapRoot::<DiffCoverage>::new([1.0f64; 3], [0f64; 3], 1024);

    let r = 6.0;

    println!("Naive");
    let now = Instant::now();
    for p in &points {
        hmap.add_cov(&Sphere::new(*p, r), &mut |_| ());
    }
    println!("    End add {:?}", now.elapsed());
    let now = Instant::now();
    let mut w = 0;
    for p in &points {
        w += hmap.del_cov(&Sphere::new(*p, r), &mut |_| ());
    }
    println!("    End del {:?}", now.elapsed());

    println!("CovMap");
    println!("    Start add");
    let now = Instant::now();
    let mut m = 0;
    for p in &points {
        m += map.add_cov(&Sphere::new(*p, r), &mut |_| ());
    }
    println!("    End add {:?}", now.elapsed());

    println!("DiffMap");
    println!("    Start diff add");
    let now = Instant::now();
    let mut d = 0;
    for p in &points {
        d += dmap.add_cov(&Sphere::new(*p, r), &mut |_| ());
    }
    println!("    End add {:?}", now.elapsed());

    println!("    Start diff del");
    let now = Instant::now();
    let mut dd = 0;
    for p in &points {
        dd += dmap.del_cov(&Sphere::new(*p, r), &mut |_| ());
    }
    println!("    End diff {:?}", now.elapsed());

    println!(
        "NaiveDiffMap add: {}, CoverageMap add: {}, DiffMap add: {}, DiffMap del: {}",
        w, m, d, dd
    );
    assert!(m == d && m == dd && w == dd, "Oops, numbers don't match");
}
