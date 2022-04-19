(set-info :smt-lib-version 2.6)
(set-logic UFDT)
(set-info :category "crafted")
(declare-datatypes ((nat 0) (lst 0)) (((s (s0 nat)) (zero)) ((nil) (cons (cons0 nat) (cons1 lst)))))
(define-fun-rec snoc
  ((xs lst) (x nat)) lst
  (match xs
    ((nil (cons x nil))
     ((cons y ys) (cons y (snoc ys x))))))

(define-fun-rec rev
  ((xs lst)) lst
  (match xs
    ((nil nil)
     ((cons y ys) (snoc (rev ys) y)))))

(define-fun-rec even
  ((n nat)) Bool
  (match n
    ((zero true)
     ((s x) (match x
             ((zero false)
              ((s m) (even m))))))))

(define-fun-rec rev_n
  ((n nat) (xs lst)) lst
  (match n
    ((zero xs)
     ((s x) (rev (rev_n x xs))))))

(assert (not (forall ((n nat) (xs lst)) (=> (even n) (= (rev_n n xs) xs)))))
(check-sat)
