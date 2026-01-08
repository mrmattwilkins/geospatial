use geo::{LineString, Polygon};
use geospatial::line_before_poly;

fn main() {
    let line = LineString::from(vec![(0.5, 3.0), (0.5, 0.5)]);
    let poly = Polygon::new(LineString::from(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]), vec![]);
    println!("Bit of line {:?} before poly {:?} is {:?}", line, poly, line_before_poly(&line, &poly));

    let line = LineString::from(vec![(0.5, 3.0), (3.5, 1.5)]);
    let poly = Polygon::new(LineString::from(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]), vec![]);
    println!("Bit of line {:?} before poly {:?} is {:?}", line, poly, line_before_poly(&line, &poly));

}

