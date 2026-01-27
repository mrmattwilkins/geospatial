use geo::{Polygon};
use geospatial::{edges_to_linestring, marching_squares};
use ndarray::array;

fn main() {
    let grid = array![[0, 0, 1], [1, 1, 0]];
    let e = marching_squares(&grid);
    let ls = edges_to_linestring(1, &e[&1], &grid);

    let grid = array![[1, 1, 0, 1], [1, 0, 1, 0], [1, 1, 1, 0]];
    let e = marching_squares(&grid);
    let ls = edges_to_linestring(1, &e[&1], &grid);


    //println!("{:?}", ls);
    /*
    let grid = array![[1, 0, 0, 1], [2, 1, 0, 1], [0, 2, 1, 1],];
    let e = marching_squares(&grid);
    let ls = edges_to_linestring(1, &e[&1], &grid);
    println!("{:?}", ls);

    let ls = edges_to_linestring(2, &e[&2], &grid);
    println!("{:?}", ls);
    */
}
