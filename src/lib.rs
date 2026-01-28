//! # Geospatial
//!
//! `geospatial` provides functions for geospatial manipulation that I could not
//! find in any other rust crate.
//!

use geo::line_intersection::{LineIntersection, line_intersection};
use geo::{Coord, CoordNum, LineString, MultiLineString, Polygon};
use line_drawing::{SignedNum, Supercover};
use ndarray::Array2;
use std::collections::{HashMap, HashSet};
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
/// Extracts oriented boundary edges from a 2d array.  A horizontal or vertical edge exists between
/// two cells if they have
/// different values. It is intended for
/// grids containing region or watershed labels, where each distinct value represents
/// a separate area and you want to get the boundary edges.  The edges are oriented so the inside
/// is to the left
///
/// # Parameters
///
/// - `grid`: A 2D array of values representing labeled regions.
///
/// # Returns
///
/// A HashMap mapping each unique grid value to a list of oriented edges `(Coord<usize>, Coord<usize>)`
/// associated with that region.  The grid value is to the left of each edge.
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
///         (Coord{ x: 1, y: 0}, Coord{ x: 0, y: 0}),
///         (Coord{ x: 0, y: 1}, Coord{ x: 1, y: 1}),
///         (Coord{ x: 0, y: 0}, Coord{ x: 0, y: 1}),
///         (Coord{ x: 1, y: 1}, Coord{ x: 1, y: 0}),
///     ]
/// );
///
/// let grid = array![
///     [1, 1],
/// ];
/// assert_eq!(geospatial::marching_squares(&grid)[&1],
///     vec![
///         (Coord{ x: 1, y: 0}, Coord{ x: 0, y: 0}),
///         (Coord{ x: 0, y: 1}, Coord{ x: 1, y: 1}),
///         (Coord{ x: 2, y: 0}, Coord{ x: 1, y: 0}),
///         (Coord{ x: 1, y: 1}, Coord{ x: 2, y: 1}),
///         (Coord{ x: 0, y: 0}, Coord{ x: 0, y: 1}),
///         (Coord{ x: 2, y: 1}, Coord{ x: 2, y: 0}),
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
///         (Coord{ x: 1, y: 0}, Coord{ x: 0, y: 0}),
///         (Coord{ x: 2, y: 0}, Coord{ x: 1, y: 0}),
///         (Coord{ x: 1, y: 2}, Coord{ x: 2, y: 2}),
///         (Coord{ x: 0, y: 0}, Coord{ x: 0, y: 1}),
///         (Coord{ x: 2, y: 1}, Coord{ x: 2, y: 0}),
///         (Coord{ x: 2, y: 2}, Coord{ x: 2, y: 1}),
///         (Coord{ x: 0, y: 1}, Coord{ x: 1, y: 1}),
///         (Coord{ x: 1, y: 1}, Coord{ x: 1, y: 2}),
///     ]
/// );
/// assert_eq!(e[&2],
///     vec![
///         (Coord{ x: 0, y: 2}, Coord{ x: 1, y: 2}),
///         (Coord{ x: 0, y: 1}, Coord{ x: 0, y: 2}),
///         (Coord{ x: 1, y: 1}, Coord{ x: 0, y: 1}),
///         (Coord{ x: 1, y: 2}, Coord{ x: 1, y: 1}),
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
///         (Coord{ x: 1, y: 0}, Coord{ x: 0, y: 0}),
///         (Coord{ x: 0, y: 0}, Coord{ x: 0, y: 1}),
///         (Coord{ x: 1, y: 1}, Coord{ x: 1, y: 0}),
///         (Coord{ x: 0, y: 1}, Coord{ x: 1, y: 1}),
///     ]
/// );
/// assert_eq!(e[&3],
///     vec![
///         (Coord{ x: 4, y: 2}, Coord{ x: 4, y: 1}),
///         (Coord{ x: 3, y: 1}, Coord{ x: 3, y: 2}),
///         (Coord{ x: 4, y: 1}, Coord{ x: 3, y: 1}),
///         (Coord{ x: 3, y: 2}, Coord{ x: 4, y: 2}),
///     ]
/// );
/// assert_eq!(e[&2],
///     vec![
///         (Coord{ x: 1, y: 3}, Coord{ x: 2, y: 3}),
///         (Coord{ x: 2, y: 3}, Coord{ x: 3, y: 3}),
///         (Coord{ x: 4, y: 0}, Coord{ x: 3, y: 0}),
///         (Coord{ x: 3, y: 3}, Coord{ x: 4, y: 3}),
///         (Coord{ x: 4, y: 1}, Coord{ x: 4, y: 0}),
///         (Coord{ x: 4, y: 3}, Coord{ x: 4, y: 2}),
///         (Coord{ x: 3, y: 0}, Coord{ x: 3, y: 1}),
///         (Coord{ x: 3, y: 1}, Coord{ x: 2, y: 1}),
///         (Coord{ x: 2, y: 1}, Coord{ x: 2, y: 2}),
///         (Coord{ x: 2, y: 2}, Coord{ x: 1, y: 2}),
///         (Coord{ x: 3, y: 2}, Coord{ x: 3, y: 1}),
///         (Coord{ x: 3, y: 1}, Coord{ x: 4, y: 1}),
///         (Coord{ x: 4, y: 2}, Coord{ x: 3, y: 2}),
///         (Coord{ x: 1, y: 2}, Coord{ x: 1, y: 3}),
///     ]
/// );
/// assert_eq!(e[&1],
///     vec![
///         (Coord{ x: 0, y: 3}, Coord{ x: 1, y: 3}),
///         (Coord{ x: 2, y: 0}, Coord{ x: 1, y: 0}),
///         (Coord{ x: 3, y: 0}, Coord{ x: 2, y: 0}),
///         (Coord{ x: 0, y: 1}, Coord{ x: 0, y: 2}),
///         (Coord{ x: 0, y: 2}, Coord{ x: 0, y: 3}),
///         (Coord{ x: 1, y: 0}, Coord{ x: 1, y: 1}),
///         (Coord{ x: 1, y: 1}, Coord{ x: 0, y: 1}),
///         (Coord{ x: 3, y: 1}, Coord{ x: 3, y: 0}),
///         (Coord{ x: 2, y: 1}, Coord{ x: 3, y: 1}),
///         (Coord{ x: 2, y: 2}, Coord{ x: 2, y: 1}),
///         (Coord{ x: 1, y: 2}, Coord{ x: 2, y: 2}),
///         (Coord{ x: 1, y: 3}, Coord{ x: 1, y: 2}),
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
        let edge = (Coord { x: c + 1, y: r }, Coord { x: c, y: r });
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
        let edge = (Coord { x: c + 1, y: r + 1 }, Coord { x: c + 1, y: r });
        ret.entry(me).or_default().push(edge);
    }

    // fill in the interior
    for r in 0..nrows - 1 {
        for c in 0..ncols - 1 {
            let me = grid[[r, c]];
            let right = grid[[r, c + 1]];
            let down = grid[[r + 1, c]];
            if me != right {
                let edge = (Coord { x: c + 1, y: r + 1 }, Coord { x: c + 1, y: r });
                ret.entry(me).or_default().push(edge);
                let edge = (Coord { x: c + 1, y: r }, Coord { x: c + 1, y: r + 1 });
                ret.entry(right).or_default().push(edge);
            }
            if me != down {
                let edge = (Coord { x: c, y: r + 1 }, Coord { x: c + 1, y: r + 1 });
                ret.entry(me).or_default().push(edge);
                let edge = (Coord { x: c + 1, y: r + 1 }, Coord { x: c, y: r + 1 });
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
            let edge = (Coord { x: c + 1, y: r + 1 }, Coord { x: c, y: r + 1 });
            ret.entry(down).or_default().push(edge);
        }
    }

    // last row, except bottom right hand cell
    for c in 0..ncols - 1 {
        let r = nrows - 1;
        let me = grid[[r, c]];
        let right = grid[[r, c + 1]];
        if me != right {
            let edge = (Coord { x: c + 1, y: r + 1 }, Coord { x: c + 1, y: r });
            ret.entry(me).or_default().push(edge);
            let edge = (Coord { x: c + 1, y: r }, Coord { x: c + 1, y: r + 1 });
            ret.entry(right).or_default().push(edge);
        }
    }

    ret
}

/// Hierholzer's algorithm to find Euler circuit from adjacency
/// adj represents the adjacency list of the directed graph
/// returns a vector of vertex indices in order
pub fn hierholzer(mut adj: Vec<Vec<usize>>) -> Option<Vec<usize>> {
    if adj.is_empty() {
        return None;
    }

    // number of outgoing edges per vertex (and indeg to just check)
    let mut edge_count: HashMap<usize, usize> = HashMap::new();
    let mut indeg: HashMap<usize, usize> = HashMap::new();
    for i in 0..adj.len() {
        edge_count.insert(i, adj[i].len());
        for &v in &adj[i] {
            *indeg.entry(v).or_insert(0) += 1;
        }
    }

    // check in deg == out deg
    for i in 0..adj.len() {
        let out = *edge_count.get(&i)?;
        let inn = *indeg.get(&i)?;
        if out != inn {
            return None;
        }
    }


    // stack of current path, and final circuit
    let mut curr_path: Vec<usize> = Vec::new();
    let mut circuit: Vec<usize> = Vec::new();

    // start anywhere
    curr_path.push(0);
    let mut curr_v: usize = 0;

    while !curr_path.is_empty() {
        let ec = edge_count.get(&curr_v)?;

        // If there's remaining edge
        if *ec > 0 {
            curr_path.push(curr_v);

            let next_v = adj[curr_v].pop()?;
            let ec_mut = edge_count.get_mut(&curr_v)?;
            *ec_mut -= 1;
            
            curr_v = next_v;
        }
        // Back track to find remaining circuit
        else {
            circuit.push(curr_v);
            curr_v = curr_path.pop()?;
        }
    }

    circuit.reverse();
    Some(circuit)
}

/// Converts a collection of unordered grid edges that form a bunch of rings into a
/// `LineString` or None if we can't.  The LineString can have repeated points, ie it can touch
/// itself, however it will not cross itself.
///
/// This function takes a list of edges, where each edge is represented by a pair
/// of grid coordinates, and converts them into a `LineString`.
/// The edges should completely encircle regions.
///
/// # Parameters
///
/// - `edges`: A vector of edge segments, where each edge is represented as a pair
///   of `Coord<usize>` values defining the start and end points.
///
/// # Returns
///
/// A Option<LineString<usize>> where input edges have been ordered to make a
/// LineStrings.
///
/// # Examples
///
/// ```
/// use geo::{Coord, LineString};
/// use ndarray::array;
///
/// let grid = array![[0, 0, 1],[1, 1, 0]];
/// let e = geospatial::marching_squares(&grid);
/// let ls = geospatial::edges_to_linestring(&e[&1]);
/// assert_eq!(ls, Some(LineString::from(vec![
///    Coord { x: 0, y: 2 },
///    Coord { x: 1, y: 2 },
///    Coord { x: 2, y: 2 },
///    Coord { x: 2, y: 1 },
///    Coord { x: 3, y: 1 },
///    Coord { x: 3, y: 0 },
///    Coord { x: 2, y: 0 },
///    Coord { x: 2, y: 1 },
///    Coord { x: 1, y: 1 },
///    Coord { x: 0, y: 1 },
///    Coord { x: 0, y: 2 },
/// ])));
/// let grid = array![[1, 1, 0, 1], [1, 0, 1, 0], [1, 1, 1, 0]];
/// let e = geospatial::marching_squares(&grid);
/// let ls = geospatial::edges_to_linestring(&e[&1]);
/// assert_eq!(ls, Some(LineString::from(vec![
///    Coord { x: 1, y: 0 },
///    Coord { x: 0, y: 0 },
///    Coord { x: 0, y: 1 },
///    Coord { x: 0, y: 2 },
///    Coord { x: 0, y: 3 },
///    Coord { x: 1, y: 3 },
///    Coord { x: 2, y: 3 },
///    Coord { x: 3, y: 3 },
///    Coord { x: 3, y: 2 },
///    Coord { x: 3, y: 1 },
///    Coord { x: 4, y: 1 },
///    Coord { x: 4, y: 0 },
///    Coord { x: 3, y: 0 },
///    Coord { x: 3, y: 1 },
///    Coord { x: 2, y: 1 },
///    Coord { x: 2, y: 2 },
///    Coord { x: 1, y: 2 },
///    Coord { x: 1, y: 1 },
///    Coord { x: 2, y: 1 },
///    Coord { x: 2, y: 0 },
///    Coord { x: 1, y: 0 },
/// ])));
/// let grid = array![[0]];
/// let e = geospatial::marching_squares(&grid);
/// let ls = geospatial::edges_to_linestring(&e[&0]);
/// assert_eq!(ls, Some(LineString::from(vec![
///    Coord { x: 1, y: 0 },
///    Coord { x: 0, y: 0 },
///    Coord { x: 0, y: 1 },
///    Coord { x: 1, y: 1 },
///    Coord { x: 1, y: 0 },
/// ])));
/// let ls = geospatial::edges_to_linestring(&vec![(Coord {x:0, y:0}, Coord {x:0, y:1})]);
/// assert_eq!(ls, None);
/// let ls = geospatial::edges_to_linestring(&vec![]);
/// assert_eq!(ls, None);
/// let ls = geospatial::edges_to_linestring(&vec![(Coord {x:0, y:0}, Coord {x:0, y:1}), (Coord {x:0, y:0}, Coord {x:0, y:2})]);
/// assert_eq!(ls, None);
/// ```
pub fn edges_to_linestring(edges: &Vec<(Coord<usize>, Coord<usize>)>) -> Option<LineString<usize>>
{
    // we will use Hierholzer's_algorithm to find Euler circle through every vertex/Coord,
    // including the knots which have index four
    
    // a vertex is Coord, keep track of Coord to vertex index
    let mut vertices: HashMap<Coord<usize>, usize> = HashMap::new();

    // adjancy matrix, the indices are vertex indices
    //      adj[i] = [j] means edge i->j
    //      adj[i] = [j, k] means edge i->j i->k
    let mut adj: Vec<Vec<usize>> = Vec::new();

    // this gets vertex ID, or makes one if necessary and enlarges adj
    let get_vid = |c: Coord<usize>, vertices: &mut HashMap<Coord<usize>, usize>, adj: &mut Vec<Vec<usize>>| -> usize
    {
        if let Some(&id) = vertices.get(&c) {
            return id;
        }
        let id = vertices.len();
        vertices.insert(c, id);
        adj.push(Vec::new());
        id
    };

    // build the adjancy
    for (from, to) in edges {
        let from_id = get_vid(*from, &mut vertices, &mut adj);
        let to_id   = get_vid(*to, &mut vertices, &mut adj);
        adj[from_id].push(to_id);
    }

    // vec of vertex indices in order
    if let Some(circuit) = hierholzer(adj) {
        // we need to reverse the vertices map to get coords from vids
        let mut vid2coord: HashMap<usize, Coord<usize>> = HashMap::new();
        for (coord, vid) in vertices {
            vid2coord.insert(vid, coord);
        }
        let ls: Vec<Coord<usize>> = circuit.into_iter().map(|vid| vid2coord.get(&vid).copied()).collect::<Option<Vec<_>>>()?;
        return Some(LineString::new(ls));
    } else {
        return None;
    }
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
///    Coord { x: 1, y: 0 },
///    Coord { x: 1, y: 1 },
///    Coord { x: 0, y: 1 },
///    Coord { x: 0, y: 0 },
///    Coord { x: 1, y: 0 },
/// ]));
/// let grid = array![
///     [0, 1],
///     [1, 1],
/// ];
/// let e = geospatial::marching_squares(&grid);
/// let mls = geospatial::edges_to_multilinestring(0, &e[&0], &grid);
/// assert_eq!(mls.0.len(), 1);
/// assert_eq!(mls.0[0], LineString::from(vec![
///    Coord { x: 1, y: 0 },
///    Coord { x: 1, y: 1 },
///    Coord { x: 0, y: 1 },
///    Coord { x: 0, y: 0 },
///    Coord { x: 1, y: 0 },
/// ]));
/// let grid = array![
///     [1, 1],
///     [1, 1],
/// ];
/// let e = geospatial::marching_squares(&grid);
/// let mls = geospatial::edges_to_multilinestring(1, &e[&1], &grid);
/// assert_eq!(mls.0.len(), 1);
/// assert_eq!(mls.0[0], LineString::from(vec![
///    Coord { x: 1, y: 0 },
///    Coord { x: 2, y: 0 },
///    Coord { x: 2, y: 1 },
///    Coord { x: 2, y: 2 },
///    Coord { x: 1, y: 2 },
///    Coord { x: 0, y: 2 },
///    Coord { x: 0, y: 1 },
///    Coord { x: 0, y: 0 },
///    Coord { x: 1, y: 0 },
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
///    Coord { x: 2, y: 0 },
///    Coord { x: 2, y: 1 },
///    Coord { x: 1, y: 1 },
///    Coord { x: 1, y: 0 },
///    Coord { x: 2, y: 0 },
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
///    Coord { x: 3, y: 2 },
///    Coord { x: 2, y: 2 },
///    Coord { x: 2, y: 1 },
///    Coord { x: 3, y: 1 },
///    Coord { x: 3, y: 2 },
/// ]));
/// ```
pub fn edges_to_multilinestring<T>(
    id: T,
    edges: &Vec<(Coord<usize>, Coord<usize>)>,
    grid: &Array2<T>,
) -> MultiLineString<usize>
where
    T: Eq + Hash + Copy,
{
    // return which two points are adjacent to our grid cell when we hit a knot
    // p is previous
    // c is where we are at
    fn adjcoords<T>(p: Coord<usize>, c: Coord<usize>, id: T, grid: &Array2<T>) -> [Coord<usize>; 2]
    where
        T: Eq + Hash + Copy,
    {
        let (row, col) = (c.y, c.x);

        // this is kinda tricky, so be explicit

        // moving right
        if p.x == c.x - 1 {
            // moving up
            if grid[[row - 1, col - 1]] == id {
                return [p, Coord { x: col, y: row - 1 }];
            }
            // moving down
            return [p, Coord { x: col, y: row + 1 }];

        // moving left
        } else if p.x == c.x + 1 {
            // moving up
            if grid[[row - 1, col]] == id {
                return [p, Coord { x: col, y: row - 1 }];
            }
            // moving down
            return [p, Coord { x: col, y: row + 1 }];

        // moving down
        } else if p.y == c.y - 1 {
            // moving left
            if grid[[row - 1, col - 1]] == id {
                return [p, Coord { x: col - 1, y: row }];
            }
            return [p, Coord { x: col + 1, y: row }];

        // moving up
        } else {
            // moving left
            if grid[[row, col - 1]] == id {
                return [p, Coord { x: col - 1, y: row }];
            }
            // moving right
            return [p, Coord { x: col + 1, y: row }];
        }
    }

    // a helper that makes a single ring.  assumes we start at a point with two neighbours
    // id and grid are used to figure out correct direction at a knot
    fn aring<T>(
        adj: &HashMap<Coord<usize>, Vec<Coord<usize>>>,
        start: Coord<usize>,
        id: T,
        grid: &Array2<T>,
    ) -> Vec<Coord<usize>>
    where
        T: Eq + Hash + Copy,
    {
        let mut ring: Vec<Coord<usize>> = Vec::new();
        let mut cur = start;
        let mut prev: Coord<usize> = adj[&cur][0];

        // storage for knot-case neighbours
        let mut knot_coords: [Coord<usize>; 2];

        loop {
            ring.push(cur);
            let mut n: &[Coord<usize>] = &adj[&cur];
            if n.len() == 4 {
                knot_coords = adjcoords(prev, cur, id, grid);
                n = &knot_coords;
            }
            if prev == n[0] && n[1] != start {
                prev = cur;
                cur = n[1];
            } else if prev == n[1] && n[0] != start {
                prev = cur;
                cur = n[0];
            } else {
                break;
            }
        }
        ring.push(start);

        return ring;
    }

    // start with a copy of edges since we will be constantly updating these
    let mut edges = edges.clone();

    let mut rings: Vec<LineString<usize>> = Vec::new();
    while edges.len() != 0 {
        // build the adjancey
        let mut adj: HashMap<Coord<usize>, Vec<Coord<usize>>> = HashMap::new();
        for (a, b) in &edges {
            adj.entry(*a).or_default().push(*b);
            adj.entry(*b).or_default().push(*a);
        }
        assert!(adj.values().all(|p| p.len() == 2 || p.len() == 4));

        // first point of first edge will do to make a ring
        let start = edges[0].0;
        let ring = aring::<T>(&adj, start, id, &grid);
        rings.push(LineString(ring.clone()));

        let myedges: HashSet<(Coord<usize>, Coord<usize>)> = ring
            .windows(2)
            .flat_map(|w| vec![(w[0], w[1]), (w[1], w[0])])
            .collect();
        edges = edges.into_iter().filter(|e| !myedges.contains(e)).collect();
    }

    return MultiLineString::new(rings);
}

/// Returns the first intersection point between line and polygon, encountered when walking along
/// line, where first means earliest along the LineStrings segment order.  For overlapping
/// (collinear) intersections, the point closest to the line segment start is returned.
///
/// # Examples
///
/// ```
/// use geo::{Coord, LineString, Polygon};
/// let line = LineString::from(vec![(0.5, 3.0), (0.5, 0.5)]);
/// let poly = Polygon::new(LineString::from(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]), vec![]);
/// let i = geospatial::first_line_poly_intersection(&line, &poly);
/// assert_eq!(i, Some(Coord {x: 0.5, y:1.0}));
///
/// let line = LineString::from(vec![(1.0, 3.0), (1.0, -0.5)]);
/// let i = geospatial::first_line_poly_intersection(&line, &poly);
/// assert_eq!(i, Some(Coord {x: 1.0, y:1.0}));
///
/// let line = LineString::from(vec![(0.0, 2.0), (0.5, 1.0), (0.5, -2.0)]);
/// let i = geospatial::first_line_poly_intersection(&line, &poly);
/// assert_eq!(i, Some(Coord {x: 0.5, y:1.0}));
///
/// let line = LineString::from(vec![(0.0, -2.0), (0.0, 0.0), (1.0, 2.0)]);
/// let i = geospatial::first_line_poly_intersection(&line, &poly);
/// assert_eq!(i, Some(Coord {x: 0.0, y:0.0}));
///
/// let line = LineString::from(vec![(0.25, 0.0), (0.75, 0.0), (1.0, 2.0)]);
/// let i = geospatial::first_line_poly_intersection(&line, &poly);
/// assert_eq!(i, Some(Coord {x: 0.25, y:0.0}));
///
/// let line = LineString::from(vec![(0.75, 3.0), (0.75, -1.0)]);
/// let poly = Polygon::new(LineString::from(vec![(0.0, 0.0), (1.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]), vec![]);
/// let i = geospatial::first_line_poly_intersection(&line, &poly);
/// assert_eq!(i, Some(Coord {x: 0.75, y:1.0}));
///
/// let line = LineString::from(vec![(-1.0, 0.5), (2.0, 0.5)]);
/// let i = geospatial::first_line_poly_intersection(&line, &poly);
/// assert_eq!(i, Some(Coord {x: 0.0, y:0.5}));
///
/// let line = LineString::from(vec![(-1.0, 1.5), (2.0, 1.5)]);
/// let i = geospatial::first_line_poly_intersection(&line, &poly);
/// assert_eq!(i, None);
/// ```
pub fn first_line_poly_intersection(linestr: &LineString<f64>, poly: &Polygon<f64>) -> Option<Coord<f64>>
{
    let ring = poly.exterior();

    for line in linestr.lines() {
        // get intersections, there could be more than one crossing, get them all
        let ints: Vec<Coord<f64>> = ring.lines().flat_map(|seg|
            match line_intersection(line, seg) {
                Some(LineIntersection::SinglePoint { intersection, .. }) => vec![intersection],
                Some(LineIntersection::Collinear { intersection }) => vec![intersection.start, intersection.end],
                None => Vec::new()
            }
        ).collect();

        // return the closest crossing to line.start (if there is one)
        let closest: Option<Coord<f64>> = ints.iter().min_by(|a, b| {
            let da = (a.x - line.start.x).powi(2) + (a.y - line.start.y).powi(2);
            let db = (b.x - line.start.x).powi(2) + (b.y - line.start.y).powi(2);
            da.partial_cmp(&db).unwrap()
        }).copied();
        if let Some(p) = closest {
            return Some(p);
        }
    }
    return None;
}

/// Returns the first part of a line before it hits polygon ring.  This is the linestring from
/// start to that returned by first_line_poly_intersection.
///
/// # Examples
///
/// ```
/// use geo::{Coord, LineString, Polygon};
/// let line = LineString::from(vec![(0.5, 3.0), (0.5, 0.5)]);
/// let poly = Polygon::new(LineString::from(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]), vec![]);
/// let l = geospatial::line_before_poly(&line, &poly);
/// assert_eq!(l, LineString::from(vec![(0.5, 3.0), (0.5, 1.0)]));
///
/// let line = LineString::from(vec![(0.0, 8.0), (1.0, 3.0), (1.0, -2.0)]);
/// let l = geospatial::line_before_poly(&line, &poly);
/// assert_eq!(l, LineString::from(vec![(0.0, 8.0), (1.0, 3.0), (1.0, 1.0)]));
///
/// let line = LineString::from(vec![(0.0, 2.0), (0.5, 1.0), (0.5, -3.0)]);
/// let l = geospatial::line_before_poly(&line, &poly);
/// assert_eq!(l, LineString::from(vec![(0.0, 2.0), (0.5, 1.0)]));
///
/// let line = LineString::from(vec![(0.5, 1.0), (0.5, -3.0)]);
/// let l = geospatial::line_before_poly(&line, &poly);
/// assert_eq!(l, LineString::new(Vec::<Coord<f64>>::new()));
///
/// let line = LineString::from(vec![(0.5, 2.0), (1.5, 3.0)]);
/// let l = geospatial::line_before_poly(&line, &poly);
/// assert_eq!(l, LineString::from(vec![(0.5, 2.0), (1.5, 3.0)]));
///
/// let line = LineString::from(Vec::<Coord>::new());
/// let l = geospatial::line_before_poly(&line, &poly);
/// assert_eq!(l, LineString::from(Vec::<Coord<f64>>::new()));
/// ```
pub fn line_before_poly(linestr: &LineString<f64>, poly: &Polygon<f64>) -> LineString<f64>
{
    let ring = poly.exterior();
    let mut pts: Vec<Coord<f64>> = Vec::new();

    if let Some(first) = linestr.0.first() {
        pts.push(*first);
    }

    for line in linestr.lines() {

        // get intersections, there could be more than one crossing, get them all
        let ints: Vec<Coord<f64>> = ring.lines().flat_map(|seg|
            match line_intersection(line, seg) {
                Some(LineIntersection::SinglePoint { intersection, .. }) => vec![intersection],
                Some(LineIntersection::Collinear { intersection }) => vec![intersection.start, intersection.end],
                None => Vec::new(),
            }
        ).collect();

        // possibly multiple intersections, return the one closest to line.start
        let closest: Option<Coord<f64>> = ints.iter().min_by(|a, b| {
                let da = (a.x - line.start.x).powi(2) + (a.y - line.start.y).powi(2);
                let db = (b.x - line.start.x).powi(2) + (b.y - line.start.y).powi(2);
                da.partial_cmp(&db).unwrap()
        }).copied();

        if let Some(p) = closest {
            pts.push(p);
            break;
        } else {
            pts.push(line.end);
        }
    }
    // handle case the very first point was on the poly, so two duplicate points
    if pts.len() == 2 && pts[0] == pts[1] {
        return LineString::new(Vec::<Coord>::new());
    }

    return LineString::from(pts);
}
