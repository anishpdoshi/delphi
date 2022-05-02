(set-logic ALL)

; Declare parameters
(declare-fun length () Int)
(declare-fun width () Int)
(declare-fun height () Int)
(declare-fun noseLength () Int)
(declare-fun radius () Int)
(declare-fun tailLength () Int)
(declare-fun endRadius () Int)
(declare-fun payload_x () Int)
(declare-fun payload_y () Int)
(declare-fun payload_z () Int)
(declare-fun vol () Int)
(declare-fun payload_vol () Int)
(declare-fun bat_vol () Int)
(declare-fun bat_threshold () Int)

; CFD oracle
(declare-oracle-fun cfd run_surrogate (Int Int Int Int Int Int Int) Real)
(assert (>= 45.0 (cfd length width height noseLength radius tailLength endRadius)))
; Volume oracle
(declare-oracle-fun get_vol get_vol (Int Int Int Int Int Int Int) Int)

(assert (> length 10))
(assert (> width 10))
(assert (> height 10))
(assert (> noseLength 10))
(assert (> radius 1))
(assert (> tailLength 10))
(assert (> endRadius 10))
(assert (<= length 10000))
(assert (<= width 10000))
(assert (<= height 10000))
(assert (<= noseLength 10000))
(assert (<= radius 10000))
(assert (<= tailLength 10000))
(assert (<= endRadius 10000))

(assert (> length (* 2 radius)))
(assert (> width (* 2 radius)))
(assert (> height (* 2 radius)))

; Payload dimensions
; TODO: can be other shapes
(assert (= payload_x 1016))
(assert (= payload_y 1016))
(assert (= payload_z 300))
; (assert (= payload_vol (* payload_x payload_y payload_z)))
(assert (= payload_vol (* 508 508 3 payload_z)))

; Vehicle contains payload
; Can be replaced by an oracle
; (define-oracle-fun contain_payload)
(assert (or
         (and (>= length payload_x) (>= width payload_y) (>= height payload_z))
         (and (>= length payload_x) (>= width payload_z) (>= height payload_y))
         (and (>= length payload_y) (>= width payload_x) (>= height payload_z))
         (and (>= length payload_y) (>= width payload_z) (>= height payload_x))
         (and (>= length payload_z) (>= width payload_x) (>= height payload_y))
         (and (>= length payload_z) (>= width payload_y) (>= height payload_x))
         ))

; battery constraint
(assert (= bat_threshold 100000000))
(assert (= vol (get_vol length width height noseLength radius tailLength endRadius)))
(assert (= bat_vol (- vol payload_vol)))
(assert (>= bat_vol bat_threshold))

(check-sat)
(get-model)
