mod bandits;
mod random;
mod tools;


fn main() {
    bandits::cmp_algos(5, 10, 100);
    bandits::cmp_algos(5, 100, 100);
    bandits::cmp_algos(5, 1000, 100);
    bandits::cmp_algos(5, 10000, 100);
}
