use geospatial::rasterize_linestring;
use geo::{Coord, LineString};

fn main() {
    let ls: LineString<isize> = LineString::new(vec![
        Coord { x: 0, y: 0 },
        Coord { x: 2, y: 0 },
        Coord { x: 2, y: 2 },
        Coord { x: 0, y: 0 }
    ]);
    let rls = rasterize_linestring(&ls);
    println!("ls={:?} rls={:?}", ls, rls);

    let ls: LineString<i32> = LineString::new(vec![
        Coord { x: 0, y: 0 },
        Coord { x: 2, y: 0 },
        Coord { x: 2, y: 3 },
        Coord { x: 0, y: 0 }
    ]);
    let rls = rasterize_linestring(&ls);
    println!("ls={:?} rls={:?}", ls, rls);
}

