;(set-logic QF_LIA)

; Looks like Z3 has a built-in 'List' type with, at least, the following properties/methods:
; List<T>
;  head : T
;  tail : List<T>
;  insert T List<T> : List<T>

; an enumeration datatype
(declare-datatype Color ( A B C D E F ))
(declare-datatype Position ( p0 p1 p2 p3 ))

(declare-const pos_nil (List Position))
(assert (= pos_nil (as nil (List Position))))

(declare-const color_nil (List Color))
(assert (= color_nil (as nil (List Color))))

(define-fun-rec ColorListLen ((x!0 (List Color))) Int
    (ite (= x!0 (as nil (List Color))) 0 (+ 1 (ColorListLen (tail x!0))))
)
;(assert (= (ColorListLen color_nil) 0))
;(assert (= (ColorListLen (insert A color_nil)) 1))

; Returns the length of a Position list
(define-fun-rec PosListLen ((x!0 (List Position))) Int
    (ite (= x!0 pos_nil) 0 (+ 1 (PosListLen (tail x!0))))
)

; If time allows, it would be nice to automatically generate the responses to a guess
; based upon the "true" answer.
; === target = [ A B C D ]
(declare-const target (List Color))
(assert (= target (insert D (insert C (insert B (insert A (as nil (List Color))))))))

; === positions = [ 0 1 2 3 ]
(declare-const positions (List Position))
(assert (= positions (insert p3 (insert p2 (insert p1 (insert p0 pos_nil))))))

; Removes an element from a list
(define-fun-rec RemoveElement ((x!0 Int) (x!1 (List Position))) (List Position)
    (ite
        ; Are we supposed to remove the head?
        (= x!0 0)
        ; If so, return 'tail x!1'
        (tail x!1)
        ; Otherwise, return head prepended to tail with (pos-1) removed
        (insert (head x!1) (RemoveElement (- x!0 1) (tail x!1)))
    )
)
; Tests of element removal
(assert (= 
    (RemoveElement 0 (insert p1 pos_nil))
    pos_nil
))

; More element removal tests. We need to assert the length (below) because removing
; an element from nil doesn't throw an error, it just returns nil (in the same way
; that (pred 0)=0 in lambda calculus). We could probably fix this by having the function
; return a duple ((List Position) Bool) so that we could indicate success/failure, but
; this project is getting complicated enough, and, with length checks, we can work
; around this problem.
(declare-const two_elements (List Position))
(assert (=
    (RemoveElement 0 (RemoveElement 1 two_elements))
    pos_nil
))
(assert (= (PosListLen two_elements) 2))

(declare-const possible_solution (List Color))
(assert (= (ColorListLen possible_solution) 4)) ; Solution must contain 4 pegs

; Function: GuessResult
; This is the main engine for asserting implications about the responses to our guesses
; Parms:
;   guess : List<Color>    ; Our guess this time
;   possible_solution : List<Color>  ; A list of colors which must have corr_pos items in common with the guess
;   corr_pos  : Int        ; Number of stones in our guess with correct color AND position
;
; Semantics and possible sanity checks:
; corr_pos + cor_col should never be greater than len(positions)
; 'guess' contains just the colors, not positions, as the colors are for positions 0 - 3, in order. 
(define-fun-rec GuessResult ((guess (List Color)) (possible_solution (List Color)) (corr_pos Int)) Bool
    (ite
        ; First, if guess is empty, then we should not recurse
        (= 0 (ColorListLen guess))
        (ite
            ; If corr_pos == 0 (when guess is empty), then this is possibly a solution, otherwise, not
            (= corr_pos 0)
            true
            false
        )
        ; Below here, guess still has some elements
        ; EITHER this position matches OR this position doesn't match
        (or
            ; If this position matches, then the heads match AND the rest has one fewer correct pegs
            (and
                (= (head guess) (head possible_solution))
                (GuessResult (tail guess) (tail possible_solution) (- corr_pos 1))
            )
            ; If this position doesn't match, then the rest has to account for the remaining correct pegs
            (and
                (not (= (head guess) (head possible_solution)))
                (GuessResult (tail guess) (tail possible_solution) corr_pos)
            )
        )
    ) 
)

; Our secret target solution is: C A F E

; Cut here, and Z3 will suggest A A A A

; Guess A A A A with one match (position 2)
(declare-const guess1 (List Color))
(assert (= guess1 (insert A (insert A (insert A (insert A (as nil (List Color))))))))
(assert (GuessResult guess1 possible_solution 1))

; Cut here, and Z3 will suggest A B B B

; Guess A B B B with zero matches
(declare-const guess2 (List Color))
(assert (= guess2 (insert A (insert B (insert B (insert B (as nil (List Color))))))))
(assert (GuessResult guess2 possible_solution 0))

; Cut here, and Z3 will suggest B C A C

; Guess B C A C with zero matches
(declare-const guess3 (List Color))
(assert (= guess3 (insert B (insert C (insert A (insert C (as nil (List Color))))))))
(assert (GuessResult guess3 possible_solution 0))

; Cut here, and Z3 will suggest C D C A

; Guess C D C A with one match (position 1)
(declare-const guess4 (List Color))
(assert (= guess4 (insert C (insert D (insert C (insert A (as nil (List Color))))))))
(assert (GuessResult guess4 possible_solution 1))

; Cut here, and Z3 will suggest D A C D

; Guess D A C D with one match (position 2)
(declare-const guess5 (List Color))
(assert (= guess5 (insert D (insert A (insert C (insert D (as nil (List Color))))))))
(assert (GuessResult guess5 possible_solution 1))

; Cut here, and Z3 will suggest C A D E

; Guess C A D E with three matches (positions 1, 2, 4)
(declare-const guess6 (List Color))
(assert (= guess6 (insert C (insert A (insert D (insert E (as nil (List Color))))))))
(assert (GuessResult guess6 possible_solution 3))

; Cut here, and Z3 will suggest C A D F

; Guess C A D F with two matches (positions 1, 2)
(declare-const guess7 (List Color))
(assert (= guess7 (insert C (insert A (insert D (insert F (as nil (List Color))))))))
(assert (GuessResult guess7 possible_solution 2))

; Cut here, and Z3 will suggest C A E E

; Guess C A E E with three matches (positions 1, 2, 4)
(declare-const guess8 (List Color))
(assert (= guess8 (insert C (insert A (insert E (insert E (as nil (List Color))))))))
(assert (GuessResult guess8 possible_solution 3))

; Z3 suggests C A F E   Yay!

(check-sat)
(get-value (guess1 guess2 guess3 guess4 guess5 guess6 guess7 guess8 possible_solution))
