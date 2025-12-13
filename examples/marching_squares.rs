use ndarray::array;
use geospatial::marching_squares;

fn main() {
    let grid = array![
        [4, 1, 1, 2],
        [1, -1, 2, 3],
        [1, 2, 2, -1],
    ];
    let e = marching_squares(&grid);
    println!("grid = {:?}\nedges are {:?}", grid, e);
}

