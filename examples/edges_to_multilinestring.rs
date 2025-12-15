use ndarray::array;
use geospatial::{edges_to_multilinestring, marching_squares};

fn main() {
    let grid = array![
        [1, 0, 0, 1],
        [2, 1, 0, 1],
        [0, 2, 1, 1],
    ];
    let e = marching_squares(&grid);
    let mls = edges_to_multilinestring(1, &e[&1], &grid);
    println!("{:?}", mls);
    let mls = edges_to_multilinestring(2, &e[&2], &grid);
    println!("{:?}", mls);
}
