use std::fs::File;
use std::io::Write;
use std::time::Instant;
use voxcov::coverage::{
    AddCoverage, Coverage, DiffCoverage, MapLike, MapRoot, NaiveDiffCoverageMap,
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

fn bench_loop<M: MapLike + AddCoverage>(points: &[[f64; 3]], q: &[usize], file: &str) {
    let mut w = File::create(file).unwrap();
    for r in q {
        let mut map = M::new([1.0f64; 3], [0f64; 3], 1024);
        let mut z = 0;
        let now = Instant::now();
        for p in points {
            z += map.add_cov(&Sphere::new(*p, *r as f64), &mut |_| ());
        }
        let t = now.elapsed().as_millis();
        writeln!(&mut w, "{}, {}, {}", r, z, t).unwrap();
        println!("{} {} {} T({})", file, r, z, t);
    }
}
fn uniq_vs_total(points: &[[f64; 3]], q: &[usize], file: &str) {
    let mut w = File::create(file).unwrap();
    for r in q {
        let mut map = NaiveDiffCoverageMap::new([1.0f64; 3], [0f64; 3], 1024);
        let mut z = 0;
        for p in points {
            z += map.add_cov(&Sphere::new(*p, *r as f64), &mut |_| ());
        }
        writeln!(&mut w, "{}, {}", r, z).unwrap();
        println!("{} {} {}", file, r, z);
    }
}

fn main() {
    let stream = include_str!("data/4ki8.xyz");
    let points = read_points_xyz(stream);
    let q = vec![
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    ];
    bench_loop::<MapRoot<Coverage>>(&points, &q, "bench_cov.csv");
    bench_loop::<MapRoot<DiffCoverage>>(&points, &q, "bench_diff.csv");
    bench_loop::<NaiveDiffCoverageMap>(&points, &q, "bench_naive.csv");
    uniq_vs_total(&points, &q, "uniq_vs_total.csv");
}
