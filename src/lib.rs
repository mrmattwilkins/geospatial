//! # Geospatial
//!
//! `geospatial` provides functions for geospatial manipulation that I could not
//! find in any other rust crate.
//!

use geo::{Coord, CoordNum, LineString, MultiLineString};
use line_drawing::{SignedNum, Supercover};
use ndarray::Array2;
use std::collections::{HashSet, HashMap};
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
/// use geo::{Coord, MultiLineString, LineString};
/// use ndarray::array;
///
/// let grid = array![[0]];
/// let e = geospatial::marching_squares(&grid);
/// let mls = geospatial::edges_to_multilinestring(0, &e[&0], &grid);
/// assert_eq!(mls.0.len(), 1);
/// assert_eq!(mls.0[0], LineString::from(vec![
///    Coord { x: 0, y: 0 },
///    Coord { x: 0, y: 1 },
///    Coord { x: 1, y: 1 },
///    Coord { x: 1, y: 0 },
///    Coord { x: 0, y: 0 },
/// ]));
/// let grid = array![
///     [0, 1],
///     [1, 1],
/// ];
/// let e = geospatial::marching_squares(&grid);
/// let mls = geospatial::edges_to_multilinestring(0, &e[&0], &grid);
/// assert_eq!(mls.0.len(), 1);
/// assert_eq!(mls.0[0], LineString::from(vec![
///    Coord { x: 0, y: 0 },
///    Coord { x: 0, y: 1 },
///    Coord { x: 1, y: 1 },
///    Coord { x: 1, y: 0 },
///    Coord { x: 0, y: 0 },
/// ]));
/// let grid = array![
///     [1, 1],
///     [1, 1],
/// ];
/// let e = geospatial::marching_squares(&grid);
/// let mls = geospatial::edges_to_multilinestring(1, &e[&1], &grid);
/// assert_eq!(mls.0.len(), 1);
/// assert_eq!(mls.0[0], LineString::from(vec![
///    Coord { x: 0, y: 0 },
///    Coord { x: 0, y: 1 },
///    Coord { x: 0, y: 2 },
///    Coord { x: 1, y: 2 },
///    Coord { x: 2, y: 2 },
///    Coord { x: 2, y: 1 },
///    Coord { x: 2, y: 0 },
///    Coord { x: 1, y: 0 },
///    Coord { x: 0, y: 0 },
/// ]));
/// let grid = array![
///     [0, 1, 0],
///     [1, 0, 1],
///     [0, 1, 0],
/// ];
/// let e = geospatial::marching_squares(&grid);
/// let mls = geospatial::edges_to_multilinestring(1, &e[&1], &grid);
/// assert_eq!(mls.0.len(), 4);
/// assert_eq!(mls.0[0], LineString::from(vec![
///    Coord { x: 1, y: 0 },
///    Coord { x: 1, y: 1 },
///    Coord { x: 2, y: 1 },
///    Coord { x: 2, y: 0 },
///    Coord { x: 1, y: 0 },
/// ]));
/// assert_eq!(mls.0[1], LineString::from(vec![
///    Coord { x: 1, y: 3 },
///    Coord { x: 1, y: 2 },
///    Coord { x: 2, y: 2 },
///    Coord { x: 2, y: 3 },
///    Coord { x: 1, y: 3 },
/// ]));
/// assert_eq!(mls.0[2], LineString::from(vec![
///    Coord { x: 0, y: 1 },
///    Coord { x: 1, y: 1 },
///    Coord { x: 1, y: 2 },
///    Coord { x: 0, y: 2 },
///    Coord { x: 0, y: 1 },
/// ]));
/// assert_eq!(mls.0[3], LineString::from(vec![
///    Coord { x: 3, y: 1 },
///    Coord { x: 2, y: 1 },
///    Coord { x: 2, y: 2 },
///    Coord { x: 3, y: 2 },
///    Coord { x: 3, y: 1 },
/// ]));
/// ```
pub fn edges_to_multilinestring<T>(id: T, edges: &Vec<(Coord<usize>, Coord<usize>)>, grid: &Array2<T>) -> MultiLineString<usize>
where
    T: Eq + Hash + Copy + std::fmt::Debug
{
    // return which two points are adjancent to our grid cell when we hit a knot
    // p is previous
    // c is where we are at
    fn adjcoords<T>(p: Coord<usize>, c: Coord<usize>, id: T, grid: &Array2<T>) -> [Coord<usize>; 2]
    where
        T: Eq + Hash + Copy,
    {
        let (row, col) = (c.y, c.x);

        // this is kinda tricky, so be explicit

        // moving right
        if p.x == c.x-1 {
            // moving up
            if grid[[row-1, col-1]] == id {
                return [p, Coord{x:col, y:row-1}];
            }
            // moving down
            return [p, Coord{x:col, y:row+1}];

        // moving left
        } else if p.x == c.x+1 {
            // moving up
            if grid[[row-1, col]] == id {
                return [p, Coord{x:col, y:row-1}];
            } 
            // moving down
            return [p, Coord{x:col, y:row+1}];

        // moving down
        } else if p.y == c.y-1 {
            // moving left
            if grid[[row-1, col-1]] == id {
                return [p, Coord{x:col-1, y:row}];
            }
            return [p, Coord{x:col+1, y:row}];

        // moving up
        } else {
            // moving left
            if grid[[row, col-1]] == id {
                return [p, Coord{x:col-1, y:row}];
            }
            // moving right
            return [p, Coord{x:col+1, y:row}];
        }

    }
<<<<<<< HEAD
    assert!(adj.values().all(|p| p.len() == 2 || p.len() == 4));

    return MultiLineString::new(vec![LineString::new(vec![])]);
=======
    // a helper that makes a single ring.  assumes we start at a point with two neighbours
    // id and grid are used to figure out correct direction at a knot
    fn aring<T>(adj: &HashMap<Coord<usize>, Vec<Coord<usize>>>, start: Coord<usize>, id: T, grid: &Array2<T>) -> Vec<Coord<usize>>
    where
        T: Eq + Hash + Copy + std::fmt::Debug
    {
        let mut ring: Vec<Coord<usize>> = Vec::new();
        let mut cur = start;

        // storage for knot-case neighbours
        let mut knot_coords: [Coord<usize>; 2];

        // our neighbours, if if there are four we need to do more work
        let mut n: &[Coord<usize>] = &adj[&cur];
        if n.len() == 4 {
            knot_coords = adjcoords(n[0], cur, id, grid);
            n = &knot_coords;
        }

        let mut prev: Coord<usize> = n[0];

        loop {
            //println!("id={:?}, start={:?}, prev={:?}, cur={:?}", id, start, prev, cur);
            //let row = cur.y;
            //let col = cur.x;
            //println!("{:?}", grid.slice(s![ (row-2) .. (row+2), (col-2) .. (col+2)]));
            ring.push(cur);
            // our neighbours, if if there are four we need to do more work
            n = &adj[&cur];
            //println!("n = {:?}", n);
            if n.len() == 4 {
                knot_coords = adjcoords(prev, cur, id, grid);
                n = &knot_coords;
                //println!("kn = {:?}", n);
            }
            if prev == n[0] && n[1] != start {
                prev = cur;
                cur = n[1]
            } else if prev == n[1] && n[0] != start {
                prev = cur;
                cur = n[0]
            } else {
                break;
            }
            //println!("picking next = {:?}", cur);
        }
        ring.push(start);

        return ring;
    }

    // start with a copy of edges
    let mut edges = edges.clone();

    //println!("There are {} edges so far", edges.len());

    let mut rings: Vec<LineString<usize>> = Vec::new();
    while edges.len() != 0 {
        //println!("There are {} edges so far", edges.len());

        // build the adjancey
        let mut adj: HashMap<Coord<usize>, Vec<Coord<usize>>> = HashMap::new();
        for (a, b) in &edges {
            adj.entry(*a).or_default().push(*b);
            adj.entry(*b).or_default().push(*a);
        }
        assert!(adj.values().all(|p| p.len() == 2 || p.len() == 4));

        // first point of first edge will do
        let start = edges[0].0;

        let ring = aring::<T>(&adj, start, id, &grid);
        rings.push(LineString(ring.clone()));

        let myedges: HashSet<(Coord<usize>, Coord<usize>)> = ring.windows(2).flat_map(|w| vec![(w[0], w[1]), (w[1], w[0])]).collect();
        edges = edges.into_iter().filter(|e| !myedges.contains(e)).collect();
    }

    return MultiLineString::new(rings);
>>>>>>> 77bc0c003201a6d4c46ed823a239a892ac2a1c7f
}

