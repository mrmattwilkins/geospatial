//! # Geospatial
//!
//! `geospatial` provides functions for geospatial manipulation that I could not
//! find in any other rust crate.
//!

use geo::{Coord, CoordNum, LineString};
use line_drawing::{Supercover, SignedNum};

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
    where T: CoordNum + SignedNum
{
    let mut out = Vec::new();
    for w in ls.0.windows(2) {
        for (x, y) in Supercover::new((w[0].x, w[0].y), (w[1].x, w[1].y)) {
            let c = Coord {x, y};
            if Some(&c) != out.last() {
                out.push(c);
            }
        }
    }
    out
}

//
// use rayon::prelude::*;
// use geo::{LineString, Coord};
// use line_drawing::Supercover;
// 
// pub fn rasterize_linestring_parallel<T>(ls: &LineString<T>) -> Vec<Coord<T>>
// where
//     T: CoordNum + SignedNum + Send + Sync + Copy,
// {
//     // Process each segment in parallel
//     let mut segment_vecs: Vec<Vec<Coord<T>>> = ls.0.windows(2)
//         .collect::<Vec<_>>()  // collect windows so we can par_iter
//         .par_iter()
//         .map(|w| {
//             let mut seg_coords = Vec::new();
//             for (x, y) in Supercover::new((w[0].x, w[0].y), (w[1].x, w[1].y)) {
//                 seg_coords.push(Coord { x, y });
//             }
//             seg_coords
//         })
//         .collect();
// 
//     // Flatten segments into one Vec, removing duplicates at segment boundaries
//     let mut out = Vec::new();
//     for seg in segment_vecs.drain(..) {
//         if !out.is_empty() && !seg.is_empty() {
//             // Remove first cell if it duplicates last of previous segment
//             if out.last() == Some(&seg[0]) {
//                 out.extend_from_slice(&seg[1..]);
//                 continue;
//             }
//         }
//         out.extend_from_slice(&seg);
//     }
// 
//     out
// }
// 
