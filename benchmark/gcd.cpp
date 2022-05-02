#include <iostream>
#include <string>
#include <cassert>

int gcd(int a, int b) {
    if (a <= 0 || b <= 0) return 0;
    if (a == b) return a;
    if (a > b) return gcd(a - b, b);
    else return gcd(a, b - a);
}

int main(int argc, char ** argv) {
    assert(argc == 3);
    int n1 = std::atoi(argv[1]);
    int n2 = std::atoi(argv[2]);
    std::cout << gcd(n1, n2) << '\n';
}
