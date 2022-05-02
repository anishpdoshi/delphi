(declare-oracle-fun gcd gcd (Int Int) Int)

(declare-fun n1 () Int)
(declare-fun n2 () Int)

(assert (= (gcd n1 n2) 5))
(check-sat)
(get-model)
