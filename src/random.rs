extern crate rand;

use self::rand::distributions::Normal;
use self::rand::distributions::Sample;

pub fn randn() -> f64
{
    let mut normal = Normal::new(0.0, 1.0);
    let v = normal.sample(&mut rand::thread_rng());
    v
}
