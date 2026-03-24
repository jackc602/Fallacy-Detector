from z3 import *

def check_fallacy(name, solver):
    result = solver.check()
    if result == sat:
        print(f"✗ {name}: FALLACY (sat — premises don't entail conclusion)")
    elif result == unsat:
        print(f"✓ {name}: VALID (unsat — conclusion necessarily follows)")
    else:
        print(f"? {name}: UNKNOWN")
    print()


# ============================================================
# 1. Modus Ponens
#    "If it rains, the ground is wet. It is raining.
#     Therefore, the ground is wet."
#    Premise 1: Rain → WetGround
#    Premise 2: Rain
#    Conclusion: WetGround
# ============================================================
Rain = Bool('Rain')
WetGround = Bool('WetGround')

s1 = Solver()
s1.add(Implies(Rain, WetGround))
s1.add(Rain)
s1.add(Not(WetGround))  # negated conclusion

check_fallacy("Modus Ponens", s1)


# ============================================================
# 2. Modus Tollens
#    "If it rains, the ground is wet. The ground is not wet.
#     Therefore, it is not raining."
#    Premise 1: Rain → WetGround
#    Premise 2: ¬WetGround
#    Conclusion: ¬Rain
# ============================================================
s2 = Solver()
s2.add(Implies(Rain, WetGround))
s2.add(Not(WetGround))
s2.add(Rain)  # negated conclusion (¬¬Rain = Rain)

check_fallacy("Modus Tollens", s2)


# ============================================================
# 3. Universal Syllogism (Barbara)
#    "All humans are mortal. All Greeks are human.
#     Therefore, all Greeks are mortal."
#    Premise 1: ∀x (Greek(x) → Human(x))
#    Premise 2: ∀x (Human(x) → Mortal(x))
#    Conclusion: ∀x (Greek(x) → Mortal(x))
# ============================================================
Entity = DeclareSort('Entity')
x = Const('x', Entity)

Greek = Function('Greek', Entity, BoolSort())
Human = Function('Human', Entity, BoolSort())
Mortal = Function('Mortal', Entity, BoolSort())

s3 = Solver()
s3.add(ForAll([x], Implies(Greek(x), Human(x))))
s3.add(ForAll([x], Implies(Human(x), Mortal(x))))
# Negated conclusion: some Greek that is not mortal
counterexample = Const('ce', Entity)
s3.add(Greek(counterexample))
s3.add(Not(Mortal(counterexample)))

check_fallacy("Universal Syllogism (Barbara)", s3)


# ============================================================
# 4. Disjunctive Syllogism
#    "Either the battery is dead or the switch is broken.
#     The battery is not dead. Therefore, the switch is broken."
#    Premise 1: Dead ∨ Broken
#    Premise 2: ¬Dead
#    Conclusion: Broken
# ============================================================
Dead = Bool('Dead')
Broken = Bool('Broken')

s4 = Solver()
s4.add(Or(Dead, Broken))
s4.add(Not(Dead))
s4.add(Not(Broken))  # negated conclusion

check_fallacy("Disjunctive Syllogism", s4)


# ============================================================
# 5. Hypothetical Syllogism (Chain Rule)
#    "If I study, I pass. If I pass, I graduate.
#     Therefore, if I study, I graduate."
#    Premise 1: Study → Pass
#    Premise 2: Pass → Graduate
#    Conclusion: Study → Graduate
# ============================================================
Study = Bool('Study')
Pass = Bool('Pass')
Graduate = Bool('Graduate')

s5 = Solver()
s5.add(Implies(Study, Pass))
s5.add(Implies(Pass, Graduate))
# Negated conclusion: Study ∧ ¬Graduate
s5.add(Study)
s5.add(Not(Graduate))

check_fallacy("Hypothetical Syllogism", s5)


# ============================================================
# 6. Universal Elimination (Simple)
#    "All dogs are loyal. Rex is a dog.
#     Therefore, Rex is loyal."
#    Premise 1: ∀x (Dog(x) → Loyal(x))
#    Premise 2: Dog(rex)
#    Conclusion: Loyal(rex)
# ============================================================
Animal = DeclareSort('Animal')
rex = Const('rex', Animal)
a = Const('a', Animal)

Dog = Function('Dog', Animal, BoolSort())
Loyal = Function('Loyal', Animal, BoolSort())

s6 = Solver()
s6.add(ForAll([a], Implies(Dog(a), Loyal(a))))
s6.add(Dog(rex))
s6.add(Not(Loyal(rex)))  # negated conclusion

check_fallacy("Universal Elimination", s6)


# ============================================================
# 7. Existential Introduction
#    "Rex is a dog and Rex is loyal.
#     Therefore, there exists a loyal dog."
#    Premise 1: Dog(rex) ∧ Loyal(rex)
#    Conclusion: ∃x (Dog(x) ∧ Loyal(x))
# ============================================================
s7 = Solver()
s7.add(Dog(rex))
s7.add(Loyal(rex))
# Negated conclusion: no dog is loyal
s7.add(ForAll([a], Not(And(Dog(a), Loyal(a)))))

check_fallacy("Existential Introduction", s7)


# ============================================================
# 8. Conjunction Elimination
#    "It is cold and windy. Therefore, it is cold."
#    Premise: Cold ∧ Windy
#    Conclusion: Cold
# ============================================================
Cold = Bool('Cold')
Windy = Bool('Windy')

s8 = Solver()
s8.add(And(Cold, Windy))
s8.add(Not(Cold))  # negated conclusion

check_fallacy("Conjunction Elimination", s8)