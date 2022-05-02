(define-fun-rec gcd ((a Int) (b Int)) Int
    (ite (or (<= a 0) (<= b 0))
      0
      (ite (= a b)
        a
        (ite (> a b)
          (gcd (- a b) b)
          (gcd a (- b a)))))
)

(declare-fun n1 () Int)
(declare-fun n2 () Int)

(assert (= (gcd n1 n2) 18))
(check-sat)
(get-model)
