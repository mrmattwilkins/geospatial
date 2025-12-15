# Geospatial

This is a grab bag of utility functions that I couldn't find in any other rust crate


## Example of Rasterizing a line

Rasterization of a line gives you every cell that a line passes over

```
use geo::{Coord, LineString};
use geospatial::rasterize_linestring;

let ls: LineString<isize> = LineString::new(vec![
    Coord { x: 0, y: 0 },
    Coord { x: 2, y: 0 },
    Coord { x: 2, y: 2 },
    Coord { x: 0, y: 0 }
]);
let rls = rasterize_linestring(&ls);
```

## Example of calculating marching squares

Marching squares gives you every edge that contains a region in a grid

```
use ndarray::array;
use geospatial::marching_squares;

let grid = array![
    [4, 1, 1, 2],
    [1, -1, 2, 3],
    [1, 2, 2, -1],
];
let e = marching_squares(&grid);
```

## Example of edges from marching squares into MultiLineString

```
use ndarray::array;
use geospatial::{edges_to_multilinestring, marching_squares};
use geo::MultiPolygon;

let grid = array![
    [1, 0, 0, 1],
    [2, 1, 0, 1],
    [0, 2, 1, 1],
];
let e = marching_squares(&grid);
let mls = edges_to_multilinestring(1, &e[&1], &grid);
let polygons = mls.polygonize();
let mp = MultiPolygon(polygons);
```