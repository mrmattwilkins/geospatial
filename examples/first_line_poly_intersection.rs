use geo::{LineString, Polygon};
use geospatial::first_line_poly_intersection;

fn main() {
    let line = LineString::from(vec![(0.5, 3.0), (0.5, 0.5)]);
    let poly = Polygon::new(LineString::from(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]), vec![]);
    println!("Line {:?} first intersects poly {:?} at {:?}", line, poly, first_line_poly_intersection(&line, &poly));

    let line = LineString::from(vec![(0.5, 3.0), (1.5, 2.0)]);
    let poly = Polygon::new(LineString::from(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]), vec![]);
    println!("Line {:?} first intersects poly {:?} at {:?}", line, poly, first_line_poly_intersection(&line, &poly));
}

