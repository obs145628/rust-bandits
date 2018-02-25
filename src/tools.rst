

fn argmax(v : &mut Vec<f64>) -> usize {
    let mut res : usize = 0;
    for i in 0..v.len() {
        if v[i] > v[res] {
            res = i;
        }
    }
    res
}
