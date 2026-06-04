# SKELETONS


1. Cases like C+L\*Atom\*Φ were problematic. The problem with previous simplification is that it needed to wait until the whole branch was fully resolved to realize that there was another Atom in the set of multipliers. Now, it is prohibited to add an Atom if another Atom multiplier exists.

    ~~TODO~~: The same could be applied to sums.

    UPDATE: I implemented a similar function for sums.


2. Some cases like C+C\*L\*sin(L) were pruned because it identified it could be simplified into C+L\*sin(L). The problem is that C always appears at the beginning of a multiplication; thus, if an Atom is added as a multiplier, there will always be a simplification that is not prohibited. So those cases are transformed by removing the 'C\*' component before simplifying.


3. Something similar happens to cases where 'C+C\*L' and 'C+C\*P' appear. They will be immediately transformed into 'L' and 'P' before simplification. This was implemented into function 'prune\_linear\_constants(s)' but the logic was moved into the function 'apply\_rule', which is more general (See Point 5).


4. Cases like this 'C+L\*sin(C+P\*sin(C+P\*sin(C+C\*Atom\*Φ)))\*Φ' should be avoided. Because there are already 3 unary operators being used. The second Φ cannot take the form of an atom because there's an L at the beginning, and cannot take the form of a unary operator because we reached the limit already.  

    ~~TODO~~: Before checking constraints, find all Φ symbols and do a checking similar to the one in apply\_rule(...): If within their corresponding multiplications there exists Atoms AND there are no more available unary operators, then 	 prune the tree.

    UPDATE: I implemented function check\_constraints\_mult. Now, cases like 'C+C\*sin(C+C\*sin(C+L\*sin(τ)\*Φ))' are detected and pruned early.

    ~~TODO~~: The same must be applied to sums. For example, the path 'C+C\*sin(C+P\*sin(C+L\*sin(C+C\*Φ+τ)))'. Note that τ cannot (or shouldn't) be an Atom because the previous C\*Φ (or effectively, the previous Ω) could also take the value 	  of an atom. Thus, for cases where τ is a second-generation τ (it's parent is also a τ but the grandparent isn't), the only option to avoid being absorbed is using another unary operator; however, if we have already reached 	  the maximum number of unary ops allowed, we can prune the tree early.

    UPDATE: I implemented function check\_constraints\_add. Now, cases like 'C+C\*sin(L)\*sin(C+L\*sin(τ)+τ)' are detected and pruned early.


5. Cases like 'C\*sin(L) + L' are not appearing.

   UPDATE: The problem was that a case like this C+C\*sin(L)+C\*Ψ will eventually be converted into C+C\*sin(L)+C\*L or C+C\*sin(L)+L, which will be simplified into L+C\*sin(L). That simplification will be detected as not desired by the 'simplify' function. So, I updated the function 'apply\_rule' to absorb any 'C\*', 'L\*', or 'P\*' if an 'Atom' was added, and then, to absorb any subsequent 'C+', 'L+', or 'P+' that could be absorbed. As a consequence, the function 'prune\_linear\_constants(s)' is now useless and was removed.
   After Point 5, cases like 'C\*sin(L) + L' are STILL not appearing. I noticed that the grammar rules were not correct as it was not possible to have cases like C+Ω+τ since the beginning. Notice how the tree grows:

   C+Ω

   C+C\*Φ
   
   C+C\*Ψ
   
   C+C\*Ψ\*Φ
   
   Atom
   
   C+C\*sin(τ)

   C+Atom\*Φ
   
   C+C\*sin(τ)\*Φ
   
   ~~TODO~~: Update grammar rule
   
   UPDATE: After updating the production rule, 'σ': \[ \['C', '+', 'Ω'], \['C', '+', 'Ω', '+', 'τ']], the tree grows as follows:
   
   C+Ω
   
   C+Ω+τ
   
   C+C\*Φ
   
   C+C\*Φ+τ
   
   C+C\*Ψ
   
   C+C\*Ψ\*Φ
   
   C+C\*Ψ+τ
   
   C+C\*Ψ\*Φ+τ
   
   Atom
   
   C+C\*sin(τ)
   
   C+Atom\*Φ
   
   C+C\*sin(τ)\*Φ


6. ~~**TODO**~~: Inverse operations should be avoided as soon as they are declared, instead of keep growing the tree waiting to simplify them when a fully-resolved state is reached.
   
   UPDATE: It turns out that these cases are not directly found.


7. The POWER operator should not be applied directly to L or P: POWER(L)=P and POWER(P)=P.

   ~~**TODO**~~: Directly check `expr_str` and find `pow(L)` or `pow(P)` 
   UPDATE: I did it at the end of the main loop in `pipeline.py`. I called the operator `powu` to avoid problems with the original SymPy's Power operator. 


8. **This might be arguable**, but I am considering that the inversion operator is an algebraic rational operator, not a transcendental one.
As such, it should not count towards $N_D$ or $N_U$. The same applies to POWER: $\text{POWER}(x) = x^P$ only if $P \in \mathbb{Q}$ (rational).


9. In response to the previous point, even if INV and POWER don't count towards $N_D$ or $N_U$, we should prohibited self-nesting cases; e.g. INV(C+C*INV(...))
   
   ~~**TODO**~~: Update function ```apply_rule```.  
   UPDATE: I added a block to "Prohibit self-nesting of INV and POWER"


10. Cases like this `L*inv(L)`, `P*inv(L)`, `L*inv(P)` must not happen as they would be equivalent to `C+C*inv(L)`, `P+C*inv(L)`, `C+P*inv(P)`, which are already being represented.
   
      ~~**TODO**~~: First, find out why `L*inv(L)` is not being simplified automatically.
      
      UPDATE: Because `inv` was not inherently defined in SymPy's dictionary. I added function `inv` in `reducer.py` and added it to `LOCAL_DICT`. UPDATE 2: I removed it again from the dictionary, as it creates simplification issues. It is better to keep it as a unkown function for now.

      ~~**TODO**~~: The new problem is that `simplify` does not work for cases like `C+C*inv(L)`, which simplifies incorrectly to `P` 

      UPDATE: I added a new block in function `apply_rule` that checks conflicting multipliers. 

      ~~**TODO**~~: Also, cases like `inv(L)*inv(L)`, `inv(L)*inv(P)`, `inv(P)*inv(P)` must be prohibited.
      In general, multiplications of `inv`s are discouraged because `inv(A)*inv(B)` is equal to `inv(A*B)` which is already accounted for.

      ~~**TODO**~~: Only one `inv` per addition is permitted.

      UPDATE: Done, at the end of the "Prohibit self-nesting" block inside `applyrule`

       

12. ~~**TODO**~~: Similar, there must be only one POWER operation within a multiplication


13. ~~**TODO**~~: It would be convenient to allow for only two `P`s in the expression


13. Cases like `P*powu(L*exp(P) + P) + P` must be prohibited because it{s equivalent to `powu(P*exp(P) + P) + P`, which already exists. 

      **~~TODO~~**: The solution is to avoid multiplying an `Atom` to a `powu`


14. Cases like `C+C*exp(C+C*inv(L))` were being eliminated because they automatically simplify to `C*exp(C*inv(L)) + C`

      ~~**TODO**~~: Don't allow additive placeholders C inside exp.
      
      UPDATE: Added the clause `if nt.symbol == 'τ' and (nt.parent.children[0].symbol == 'exp'): production = ['Ω']` in `pipeline.py`


15. I checked that the most complicated skeleton in `exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)` (i.e., exp(C * inv(S^2)) * inv(S))
can be generated: `C*inv(L)*exp(C*inv(P)) + C`


16. ~~**TODO**~~: Note that `exp` cannot be inside an `inv` as this can already be done if the exponent is negative


17. ~~**TODO**~~: `P*inv(P)` shouldn't be added to an atom as it is redundant.


18. To discuss: What if I include the `square`, `cube`, and `sqrt` operators in addition to `powu`, as they're very characteristic functions.
That way, when sampling, `powu` could only take exponent values greater than 3. 


19. At the end of Section 4, there should be a reference to an appendix, which lists all additional rules to help simplification

    - `P` involves exponentiating a variable, so it counts towards $N_U$
    - The power functions cannot be applied directly to `L` or `P` 
    - `L*inv(L)` -> `C + C inv(L)`. `P*inv(L)` -> `P + C inv(L)`
    - `P*inv(P)` shouldn't be added to an Atom
    - `inv` cannot be nested within another `inv`, or summed or multiplied by another `inv`
    - `POW` cannot be nested within another `POW` or power operation.
    - `exp` cannot be multiplied by another `exp` or be inside an `inv`
    - `exp` is not allowed to have an additive constant

# SAMPLING



1. Cases like $\arcsin(C_1 \cdot \sin(C_2 \cdot x + C_3) + C_4)$ could be allowed. However, the restriction that should be imposed is $|C_1| + |C_4| \leq 1$
2. Justification for nested sins (which was not allowed in previous work): 

    - Kinematics and Robotics (Forward Kinematics): In a multi-jointed robotic arm (like a triple pendulum), the position of the end-effector depends on the angles of all previous joints. You will see deep compositions, but they look like: $$\sin(\theta_1 + \sin(\theta_2 + \sin(\theta_3)))$$

    - Frequency Modulation (FM Synthesis): In complex signal processing or acoustics, you can have cascading frequency modulation (e.g., Operator 1 modulates Operator 2, which modulates Operator 3). The equation might look like:$$\sin(\omega_1 t + \beta_1 \sin(\omega_2 t + \beta_2 \sin(\omega_3 t)))$$
3. Discussion about nesting trigonometric and inverse operators:
$C_1 = 1 \implies \cos(\arccos(x)) = x$
$C_1 = 2 \implies \cos(2 \arccos(x)) = 2x^2 - 1$
$C_1 = 3 \implies \cos(3 \arccos(x)) = 4x^3 - 3x$
$\cos(\arcsin(x)) = \sqrt{1 - x^2}$
$\tan(\arcsin(x)) = \frac{x}{\sqrt{1 - x^2}}$
4. Restrict transcendental operators (exp, sin, log) so they can only accept linear or low-degree polynomial inputs of depth 1 or 2 as arguments, while explicitly banning them from taking each other as direct arguments
5. Connected to point 8 from the previous "Skeletons" section, the $\text{POWER}(x) = x^P$ operator is only allowed to sample rational values for $P$. For irrational exponent values, we already have the definition $x^P = \exp(P \cdot \ln(x))$