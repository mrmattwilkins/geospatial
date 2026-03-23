use geo::{Coord, LineString};
use geospatial::centreline_rasterize_linestring;

fn main() {
    let ls: LineString<f64> = LineString::new(vec![
        Coord { x: -3.1, y: 0.1 },
        Coord { x: 7.3, y: 2.0 },
        Coord { x: 5.6, y: -6.5 },
        Coord { x: 0.0, y: 0.0 }
    ]);
    let rls = centreline_rasterize_linestring(&ls);
    println!("ls={:?} rls={:?}", ls, rls);
}
