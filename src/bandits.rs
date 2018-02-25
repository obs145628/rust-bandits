extern crate rand;

use tools;
use random;

use self::rand::Rng;


pub struct MultiArmBandit {
    pub k: usize,
    pub arms: Vec<f64>,
    pub games_arr: Vec<usize>,
    pub games: u64,
    pub wins: u64,
}

impl MultiArmBandit {


    pub fn new(k: usize) -> MultiArmBandit {
        let mut res = MultiArmBandit {
            k,
            arms: Vec::with_capacity(k),
            games_arr: Vec::with_capacity(k),
            games: 0,
            wins: 0,
        };

        let mut rng = self::rand::thread_rng();

        for _ in 0..k
        {
            res.arms.push(rng.gen_range(0., 1.));
            res.games_arr.push(0);
        }
        
        res
    }

    pub fn pull_arm(self: &mut MultiArmBandit, n: usize) -> bool {
        let mut rng = rand::thread_rng();
        let res = rng.gen_range(0., 1.) < self.arms[n as usize];
        self.games += 1;
        self.games_arr[n] += 1;
        if res {
            self.wins += 1;
        }
        res
    }

    pub fn dump(self: &MultiArmBandit) {
        println!("{} / {}", self.wins, self.games);
        for i in 0..self.k {
            println!("{} => {} ({})", i, self.arms[i], self.games_arr[i])
        }
    }
    
}


// ### EPSILON GREEDY ###
// epsilon = 1/t

pub fn run_egreedy(k : usize, npulls : u64) -> u64 {

    let mut estims : Vec<f64> = Vec::with_capacity(k);
    for _ in 0..k {
        estims.push(0.);
    }
    let mut mab = MultiArmBandit::new(k);

    let mut rng = self::rand::thread_rng();
    
    for t in 1..npulls+1 {
        let eps = 1.0 / (t as f64);
        let arm = if rng.gen_range(0., 1.) < eps {
            rng.gen_range(0, k) as usize
        }
        else {
            tools::argmax(&mut estims)
        };
        let rew = if mab.pull_arm(arm) {1.} else {0.};
        let n = mab.games_arr[arm] as f64;
        estims[arm as usize] = (1. - 1. / n) * estims[arm] + (1. / n) * rew;
    }

    println!("best arm: {}", tools::argmax(&mut estims));
    for i in 0..k {
            println!("{} => {}", i, estims[i as usize])
    }

    mab.dump();
    mab.wins
}

pub fn simu_egreedy(k : usize, npulls : u64, nsimus: u64) -> u64 {

    let mut sum = 0;
    
    for _ in 0..nsimus {
        sum += run_egreedy(k, npulls);
    }

    println!("Average: {}", (sum as f64) / (nsimus as f64));
    sum
}








// ### OPTIMAL INITIAL VALUES ###

pub fn run_oiv(k : usize, npulls : u64) -> u64 {

    let mut estims : Vec<f64> = Vec::with_capacity(k);
    for _ in 0..k {
        estims.push(2.);
    }
    let mut mab = MultiArmBandit::new(k);
    
    for _ in 0..npulls {
        let arm = tools::argmax(&mut estims);
        let rew = if mab.pull_arm(arm) {1.} else {0.};
        let n = mab.games_arr[arm] as f64;
        estims[arm as usize] = (1. - 1. / n) * estims[arm] + (1. / n) * rew;
    }

    println!("best arm: {}", tools::argmax(&mut estims));
    for i in 0..k {
            println!("{} => {}", i, estims[i as usize])
    }

    mab.dump();
    mab.wins
}

pub fn simu_oiv(k : usize, npulls : u64, nsimus: u64) -> u64 {

    let mut sum = 0;
    
    for _ in 0..nsimus {
        sum += run_oiv(k, npulls);
    }

    println!("Average: {}", (sum as f64) / (nsimus as f64));
    sum
}









// ### UCB1 ###


pub fn argmax_ucb1(v : &Vec<f64>, mab: &MultiArmBandit) -> usize {
    let mut res : usize = 0;
    let mut max_val = 0.;
    for i in 0..v.len() {

        let val = v[i] + (2. * (mab.games as f64).ln()
                          / (mab.games_arr[i] as f64 + 1e-4)).sqrt();
        
        if val > max_val {
            res = i;
            max_val = val;
        }
    }
    res
}


pub fn run_ucb1(k : usize, npulls : u64) -> u64 {

    let mut estims : Vec<f64> = Vec::with_capacity(k);
    for _ in 0..k {
        estims.push(2.);
    }
    let mut mab = MultiArmBandit::new(k);
    
    for _ in 0..npulls {
        let arm = argmax_ucb1(&estims, &mab);
        let rew = if mab.pull_arm(arm) {1.} else {0.};
        let n = mab.games_arr[arm] as f64;
        estims[arm as usize] = (1. - 1. / n) * estims[arm] + (1. / n) * rew;
    }

    println!("best arm: {}", tools::argmax(&mut estims));
    for i in 0..k {
            println!("{} => {}", i, estims[i as usize])
    }

    mab.dump();
    mab.wins
}

pub fn simu_ucb1(k : usize, npulls : u64, nsimus: u64) -> u64 {

    let mut sum = 0;
    
    for _ in 0..nsimus {
        sum += run_ucb1(k, npulls);
    }

    println!("Average: {}", (sum as f64) / (nsimus as f64));
    sum
}



// ### Thompson Sampling

struct BayeArm {
    mean: f64,
    lambda: f64,
    sum_x: f64,
    tau: f64,
}

impl BayeArm {

    fn new() -> BayeArm {
        BayeArm {
            mean: 0.,
            lambda: 1.,
            sum_x: 0.,
            tau: 1.
        }
    }

    fn sample(self: &BayeArm) -> f64 {
        return random::randn() / self.lambda.sqrt() + self.mean
    }

    fn update(self: &mut BayeArm, x : f64) {
        self.lambda += self.tau;
        self.sum_x += x;
        self.mean = (self.tau * self.sum_x) / self.lambda;
    }
    
}


fn argmax_thompson(v : &Vec<BayeArm>) -> usize {
    let mut res : usize = 0;
    let mut max_val = 0.;
    for i in 0..v.len() {

        let val = v[i].sample();
        
        if val > max_val {
            res = i;
            max_val = val;
        }
    }
    res
}


pub fn run_thompson(k : usize, npulls : u64) -> u64 {

    let mut estims : Vec<BayeArm> = Vec::with_capacity(k);
    for _ in 0..k {
        estims.push(BayeArm::new());
    }
    let mut mab = MultiArmBandit::new(k);
    
    for _ in 0..npulls {
        let arm = argmax_thompson(&estims);
        let rew = if mab.pull_arm(arm) {1.} else {0.};
        estims[arm].update(rew);
    }

    mab.dump();
    mab.wins
}

pub fn simu_thompson(k : usize, npulls : u64, nsimus: u64) -> u64 {

    let mut sum = 0;
    
    for _ in 0..nsimus {
        sum += run_thompson(k, npulls);
    }

    println!("Average: {}", (sum as f64) / (nsimus as f64));
    sum
}


pub fn cmp_algos(k: usize, npulls: u64, nsimus: u64)
{
    let v1 = simu_egreedy(k, npulls, nsimus);
    let v2 = simu_oiv(k, npulls, nsimus);
    let v3 = simu_ucb1(k, npulls, nsimus);
    let v4 = simu_thompson(k, npulls, nsimus);

    println!("Bandits: {} ; {} ; {}", k, npulls, nsimus);
    println!(" egreedy: {}", v1);
    println!("     oiv: {}", v2);
    println!("    ucb1: {}", v3);
    println!("thompson: {}", v4);
}
