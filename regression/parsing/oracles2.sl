(set-logic LIA)

(synth-fun max2 ((x Int) (y Int)) Int
    ((Start Int) (StartBool Bool))
    ((Start Int (x y 0 1 (+ Start Start) (- Start Start) (ite StartBool Start Start)))
    (StartBool Bool ((and StartBool StartBool) (or StartBool StartBool) (not StartBool) (<= Start Start) (= Start Start) (>= Start Start)))))

(declare-var x Int)
(declare-var y Int)
(declare-var w (-> Int Int Int))
(constraint (>= (max2 x y) x))
(constraint (>= (max2 x y) y))
(constraint (or (= x (max2 x y)) (= y (max2 x y))))
(constraint (= (w 1 2)(max2 1 2)))
(oracle-constraint oracle ((x Int)(y Int)) ((z Int))
(= x z)
)

(check-synth)

