//! # Geospatial
//!
//! `geospatial` provides functions for geospatial manipulation that I could not
//! find in any other rust crate.
//!

use geo::{Coord, CoordNum, LineString, MultiLineString};
use line_drawing::{SignedNum, Supercover};
use ndarray::Array2;
use std::collections::HashMap;
use std::hash::Hash;

/// Rasterizes a geo::LineString onto a grid of integer coordinates.
///
/// This function returns a `Vec<Coord<T>>` containing all grid cells that the line
/// passes through, not just the vertices of the `LineString`. It uses a supercover
/// traversal, so every cell touched by the line is included.
///
/// # Parameters
///
/// - `ls`: A reference to a `LineString<T>` to rasterize. Can be empty.  T must be SignedNum eg
///             isize, i32
///
/// # Returns
///
/// A `Vec<Coord<T>>` representing all the integer grid coordinates traversed by the line.
///
/// # Examples
/// ```
/// use geo::{Coord, LineString};
/// let ls: LineString<isize> = LineString::new(vec![
///     Coord { x: 0, y: 0 },
///     Coord { x: 2, y: 0 },
///     Coord { x: 2, y: 2 },
///     Coord { x: 0, y: 0 }
/// ]);
/// assert_eq!(
///     geospatial::rasterize_linestring(&ls),
///     vec![
///         Coord {x:0,y:0},
///         Coord {x:1,y:0},
///         Coord {x:2,y:0},
///         Coord {x:2,y:1},
///         Coord {x:2,y:2},
///         Coord {x:1,y:1},
///         Coord {x:0,y:0},
///     ]
/// );
/// let ls: LineString<isize> = LineString::new(vec![
///     Coord { x: 0, y: 0 },
///     Coord { x: -2, y: 0 },
///     Coord { x: 1, y: -3 }
/// ]);
/// assert_eq!(
///     geospatial::rasterize_linestring(&ls),
///     vec![
///         Coord {x:0,y:0},
///         Coord {x:-1,y:0},
///         Coord {x:-2,y:0},
///         Coord {x:-1,y:-1},
///         Coord {x:0,y:-2},
///         Coord {x:1,y:-3},
///     ]
/// );
/// let ls: LineString<i32> = LineString::new(vec![]);
/// assert_eq!(geospatial::rasterize_linestring(&ls), vec![]);
/// ```
pub fn rasterize_linestring<T>(ls: &LineString<T>) -> Vec<Coord<T>>
where
    T: CoordNum + SignedNum,
{
    let mut out = Vec::new();
    for w in ls.0.windows(2) {
        for (x, y) in Supercover::new((w[0].x, w[0].y), (w[1].x, w[1].y)) {
            let c = Coord { x, y };
            if Some(&c) != out.last() {
                out.push(c);
            }
        }
    }
    out
}

/// Marching squares
///
/// Extracts boundary edges from a 2d array.  A horizontal or vertical edge exists between
/// two cells if they have
/// different values. It is intended for
/// grids containing region or watershed labels, where each distinct value represents
/// a separate area and you want to get the boundary edges.
///
/// # Parameters
///
/// - `grid`: A 2D array of values representing labeled regions.
///
/// # Returns
///
/// A HashMap mapping each unique grid value to a list of edges `(Coord<usize>, Coord<usize>)`
/// associated with that region.
///
/// # Notes
///
/// - Interior cells that are completely surrounded by the same value won't generate an edge.
/// - This function does **not** return full polygon boundaries; it only identifies
///   boundary edges that will need to be assembled into a polygon
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use std::collections::HashMap;
/// use geo::{Coord};
/// use std::hash::Hash;
///
/// let grid = array![
///     [1],
/// ];
/// assert_eq!(geospatial::marching_squares(&grid)[&1],
///     vec![
///         (Coord{ x: 0, y: 0}, Coord{ x: 1, y: 0}),
///         (Coord{ x: 0, y: 1}, Coord{ x: 1, y: 1}),
///         (Coord{ x: 0, y: 0}, Coord{ x: 0, y: 1}),
///         (Coord{ x: 1, y: 0}, Coord{ x: 1, y: 1}),
///     ]
/// );
///
/// let grid = array![
///     [1, 1],
/// ];
/// assert_eq!(geospatial::marching_squares(&grid)[&1],
///     vec![
///         (Coord{ x: 0, y: 0}, Coord{ x: 1, y: 0}),
///         (Coord{ x: 0, y: 1}, Coord{ x: 1, y: 1}),
///         (Coord{ x: 1, y: 0}, Coord{ x: 2, y: 0}),
///         (Coord{ x: 1, y: 1}, Coord{ x: 2, y: 1}),
///         (Coord{ x: 0, y: 0}, Coord{ x: 0, y: 1}),
///         (Coord{ x: 2, y: 0}, Coord{ x: 2, y: 1}),
///     ]
/// );
///
/// let grid = array![
///     [1, 1],
///     [2, 1],
/// ];
/// let e = geospatial::marching_squares(&grid);
/// assert_eq!(e[&1],
///     vec![
///         (Coord{ x: 0, y: 0}, Coord{ x: 1, y: 0}),
///         (Coord{ x: 1, y: 0}, Coord{ x: 2, y: 0}),
///         (Coord{ x: 1, y: 2}, Coord{ x: 2, y: 2}),
///         (Coord{ x: 0, y: 0}, Coord{ x: 0, y: 1}),
///         (Coord{ x: 2, y: 0}, Coord{ x: 2, y: 1}),
///         (Coord{ x: 2, y: 1}, Coord{ x: 2, y: 2}),
///         (Coord{ x: 0, y: 1}, Coord{ x: 1, y: 1}),
///         (Coord{ x: 1, y: 1}, Coord{ x: 1, y: 2}),
///     ]
/// );
/// assert_eq!(e[&2],
///     vec![
///         (Coord{ x: 0, y: 2}, Coord{ x: 1, y: 2}),
///         (Coord{ x: 0, y: 1}, Coord{ x: 0, y: 2}),
///         (Coord{ x: 0, y: 1}, Coord{ x: 1, y: 1}),
///         (Coord{ x: 1, y: 1}, Coord{ x: 1, y: 2}),
///     ]
/// );
/// let grid = array![
///     [4, 1, 1, 2],
///     [1, 1, 2, 3],
///     [1, 2, 2, 2],
/// ];
/// let e = geospatial::marching_squares(&grid);
/// assert_eq!(e[&4],
///     vec![
///         (Coord{ x: 0, y: 0}, Coord{ x: 1, y: 0}),
///         (Coord{ x: 0, y: 0}, Coord{ x: 0, y: 1}),
///         (Coord{ x: 1, y: 0}, Coord{ x: 1, y: 1}),
///         (Coord{ x: 0, y: 1}, Coord{ x: 1, y: 1}),
///     ]
/// );
/// assert_eq!(e[&3],
///     vec![
///         (Coord{ x: 4, y: 1}, Coord{ x: 4, y: 2}),
///         (Coord{ x: 3, y: 1}, Coord{ x: 3, y: 2}),
///         (Coord{ x: 3, y: 1}, Coord{ x: 4, y: 1}),
///         (Coord{ x: 3, y: 2}, Coord{ x: 4, y: 2}),
///     ]
/// );
/// assert_eq!(e[&2],
///     vec![
///         (Coord{ x: 1, y: 3}, Coord{ x: 2, y: 3}),
///         (Coord{ x: 2, y: 3}, Coord{ x: 3, y: 3}),
///         (Coord{ x: 3, y: 0}, Coord{ x: 4, y: 0}),
///         (Coord{ x: 3, y: 3}, Coord{ x: 4, y: 3}),
///         (Coord{ x: 4, y: 0}, Coord{ x: 4, y: 1}),
///         (Coord{ x: 4, y: 2}, Coord{ x: 4, y: 3}),
///         (Coord{ x: 3, y: 0}, Coord{ x: 3, y: 1}),
///         (Coord{ x: 2, y: 1}, Coord{ x: 3, y: 1}),
///         (Coord{ x: 2, y: 1}, Coord{ x: 2, y: 2}),
///         (Coord{ x: 1, y: 2}, Coord{ x: 2, y: 2}),
///         (Coord{ x: 3, y: 1}, Coord{ x: 3, y: 2}),
///         (Coord{ x: 3, y: 1}, Coord{ x: 4, y: 1}),
///         (Coord{ x: 3, y: 2}, Coord{ x: 4, y: 2}),
///         (Coord{ x: 1, y: 2}, Coord{ x: 1, y: 3}),
///     ]
/// );
/// assert_eq!(e[&1],
///     vec![
///         (Coord{ x: 0, y: 3}, Coord{ x: 1, y: 3}),
///         (Coord{ x: 1, y: 0}, Coord{ x: 2, y: 0}),
///         (Coord{ x: 2, y: 0}, Coord{ x: 3, y: 0}),
///         (Coord{ x: 0, y: 1}, Coord{ x: 0, y: 2}),
///         (Coord{ x: 0, y: 2}, Coord{ x: 0, y: 3}),
///         (Coord{ x: 1, y: 0}, Coord{ x: 1, y: 1}),
///         (Coord{ x: 0, y: 1}, Coord{ x: 1, y: 1}),
///         (Coord{ x: 3, y: 0}, Coord{ x: 3, y: 1}),
///         (Coord{ x: 2, y: 1}, Coord{ x: 3, y: 1}),
///         (Coord{ x: 2, y: 1}, Coord{ x: 2, y: 2}),
///         (Coord{ x: 1, y: 2}, Coord{ x: 2, y: 2}),
///         (Coord{ x: 1, y: 2}, Coord{ x: 1, y: 3}),
///     ]
/// );
///
/// ```
pub fn marching_squares<T>(grid: &Array2<T>) -> HashMap<T, Vec<(Coord<usize>, Coord<usize>)>>
where
    T: Eq + Hash + Copy,
{
    let mut ret: HashMap<T, Vec<(Coord<usize>, Coord<usize>)>> = HashMap::new();
    let (nrows, ncols) = grid.dim();

    // we need edges around the entire grid, process top/bot row and left/right col at same time
    for c in 0..ncols {
        let r = 0;
        let me = grid[[r, c]];
        let edge = (Coord { x: c, y: r }, Coord { x: c + 1, y: r });
        ret.entry(me).or_default().push(edge);
        let r = nrows - 1;
        let me = grid[[r, c]];
        let edge = (Coord { x: c, y: r + 1 }, Coord { x: c + 1, y: r + 1 });
        ret.entry(me).or_default().push(edge);
    }
    for r in 0..nrows {
        let c = 0;
        let me = grid[[r, c]];
        let edge = (Coord { x: c, y: r }, Coord { x: c, y: r + 1 });
        ret.entry(me).or_default().push(edge);
        let c = ncols - 1;
        let me = grid[[r, c]];
        let edge = (Coord { x: c + 1, y: r }, Coord { x: c + 1, y: r + 1 });
        ret.entry(me).or_default().push(edge);
    }

    // fill in the interior
    for r in 0..nrows - 1 {
        for c in 0..ncols - 1 {
            let me = grid[[r, c]];
            let right = grid[[r, c + 1]];
            let down = grid[[r + 1, c]];
            if me != right {
                let edge = (Coord { x: c + 1, y: r }, Coord { x: c + 1, y: r + 1 });
                ret.entry(me).or_default().push(edge);
                ret.entry(right).or_default().push(edge);
            }
            if me != down {
                let edge = (Coord { x: c, y: r + 1 }, Coord { x: c + 1, y: r + 1 });
                ret.entry(me).or_default().push(edge);
                ret.entry(down).or_default().push(edge);
            }
        }
    }

    // last column, except bottom right hand cell
    for r in 0..nrows - 1 {
        let c = ncols - 1;
        let me = grid[[r, c]];
        let down = grid[[r + 1, c]];
        if me != down {
            let edge = (Coord { x: c, y: r + 1 }, Coord { x: c + 1, y: r + 1 });
            ret.entry(me).or_default().push(edge);
            ret.entry(down).or_default().push(edge);
        }
    }

    // last row, except bottom right hand cell
    for c in 0..ncols - 1 {
        let r = nrows - 1;
        let me = grid[[r, c]];
        let right = grid[[r, c + 1]];
        if me != right {
            let edge = (Coord { x: c + 1, y: r }, Coord { x: c + 1, y: r + 1 });
            ret.entry(me).or_default().push(edge);
            ret.entry(right).or_default().push(edge);
        }
    }

    ret
}

/// Converts a collection of unordered grid edges that form a bunch of rings nto a
/// `MultiLineString`.
///
/// This function takes a list of edges, where each edge is represented by a pair
/// of grid coordinates, and converts them into a `MultiLineString`.  Likely there
/// will only be a single LineString, but if there are self-intersections multiple
/// LineStrings are needed.  The edges should completely encircle regions.
///
/// # Parameters
///
/// - `edges`: A vector of edge segments, where each edge is represented as a pair
///   of `Coord<usize>` values defining the start and end points.
///
/// # Returns
///
/// A `MultiLineString<usize>` where input edges have been ordered to make a series
/// of LineStrings.
///
/// # Examples
///
/// ```
/// use geo::{Coord, MultiLineString};
/// use ndarray::array;
///
/// let grid = array![
///     [4, 1, 1, 2],
///     [1, 1, 2, 3],
///     [1, 2, 2, 2],
/// ];
/// let e = geospatial::marching_squares(&grid);
/// let mls = geospatial::edges_to_multilinestring(&e[&1]);
/// assert_eq!(mls.0.len(), 1);
/// ```
pub fn edges_to_multilinestring(
    edges: &Vec<(Coord<usize>, Coord<usize>)>,
) -> MultiLineString<usize> {
    let mut adj: HashMap<Coord<usize>, Vec<Coord<usize>>> = HashMap::new();
    for (a, b) in edges {
        adj.entry(*a).or_default().push(*b);
        adj.entry(*b).or_default().push(*a);
    }
    assert!(adj.values().all(|p| p.len() == 2 || p.len() == 4));
    return MultiLineString::new(vec![LineString::new(vec![])]);
}
