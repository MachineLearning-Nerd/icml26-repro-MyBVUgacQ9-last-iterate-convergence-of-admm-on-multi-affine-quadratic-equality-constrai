"""Per-claim page content: the exact claim, its quantifiers, what was measured,
which raw file backs it, and what remains a limitation.

Kept separate from `repro.publish` so the prose is reviewable on its own.
"""

from __future__ import annotations

CLAIMS = {
    "claim1": dict(
        title="Claim 1 - Theorem 3.1 (sublinear o(1/k) + limit point)",
        slug="claim-1",
        statement=(
            "Theorem 3.1 (source line 679). *Suppose that Assumption 2.3 and "
            "Assumption 2.6 hold. If phi is L_phi-smooth, then ADMM in Algorithm 1 "
            "with rho >= max{4 L_phi^2/(mu_phi lam_min^+(Q^T Q)), "
            "4 L_phi^2/(mu_phi sqrt(lam_min^+(Q Q^T)))} converges with at least "
            "sublinear convergence rate to a stationary point (x*, z*, w*) of the "
            "augmented Lagrangian, i.e. L(x^k,z^k,w^k) - L(x*,z*,w*) in o(1/k), "
            "where for all i, x*_i in argmin_{x_i} f(x_i, x*_-i) + I_i(x_i) subject "
            "to A(x_i, x*_-i) + Q z* = 0, and z* in argmin_z phi(z) such that "
            "A(x*) + Qz = 0.*"),
        quantifiers=[
            "**Universal** over instances of eq. (1) satisfying Assumptions 2.3 and 2.6.",
            "**Universal** over rho above an explicit, computable threshold - so rho is "
            "computed from the formula here, never tuned. Multipliers 1x, 2x and 10x "
            "of the threshold are run.",
            "The rate is `o(1/k)`, **strictly** faster than `Theta(1/k)`; a test that "
            "accepted `Theta(1/k)` would not be testing this statement.",
            "A **second, separate** conclusion: the block-wise characterisation of the "
            "limit point. It is tested independently of the rate.",
        ],
        measured=[
            "`repro/rates.py` classifies the gap `|L^k - L*|`. `L*` is read from "
            "iterations strictly beyond the analysis window, so the reference is not "
            "fitted to the decay it measures.",
            "The `o(1/k)` test requires the lower end of a bootstrap CI on the "
            "power-law exponent to exceed 1, or geometric decay (which implies it).",
            "The limit-point characterisation is checked by re-solving each block's "
            "constrained minimisation exactly and independently with an interior-point "
            "solver (Clarabel) and measuring the distance to the ADMM limit point. "
            "Because `A` is affine in each single block, each of those is a strongly "
            "convex QP and the re-solve is exact.",
            "Sub-problem KKT residuals are recorded so that 'Algorithm 1 solves the "
            "sub-problems exactly' is audited rather than assumed.",
        ],
        control=(
            "The paper's own Example 2.8 (line 664, from Gao et al. 2020): "
            "`min x^2 + y^2 s.t. xy = 1`, which is eq. (1) with `Q = 0`, so "
            "Assumption 2.6 fails while every other assumption holds. The paper "
            "predicts that from `(x^0, 0, w^0)` the iterates go to `(0,0)` with "
            "`w^k -> -inf` and an infeasible limit. The control must reproduce that "
            "failure; if the pipeline reported success here it would be measuring "
            "nothing."),
        raw=["raw/claim1/claim1_results.csv", "raw/claim1/claim1_raw.json",
             "raw/claim1/claim1_negative_control.json"],
        code=["repro/claims/claim1.py", "repro/core.py", "repro/rates.py",
              "repro/analysis.py"],
        table_cols=["instance", "n_x", "n_c", "rho", "rho_multiplier",
                    "decades_of_decay", "power_alpha", "determined",
                    "little_o_1_over_k", "nash_max_block_deviation",
                    "final_violation", "subproblem_max_kkt"],
        limitations=[
            "Theorem 3.1 is universally quantified over an infinite family of "
            "instances and asserts an asymptotic rate. Finitely many finite-horizon "
            "runs are **scoped corroboration**, not a proof: they establish that the "
            "stated rate holds on every instance tested, at the rho the theorem "
            "prescribes, and that the classifier used would have detected a violation "
            "(it rejects Theta(1/k)). No claim of a proof certificate is made.",
            "The limit-point characterisation is checked at the achieved feasibility "
            "level rather than at exact feasibility, since at finite precision the "
            "exactly-feasible block problem can be infeasible. What is tested is "
            "therefore optimality of x*_i among feasible points, which is the "
            "statement's content.",
        ],
    ),
    "claim2": dict(
        title="Claim 2 - Theorem 3.2 (equation 4 and the linear rate)",
        slug="claim-2",
        statement=(
            "Theorem 3.2 (source line 798). *Suppose that the assumptions of Theorem "
            "3.1 hold. Moreover, let f be L_f-smooth and L(x,z,w) be second-order "
            "differentiable at the limit point. When matrix Q satisfies "
            "||C|| in O(||(QQ^T)^{-1}Q||^{-1} min{m_1, m_2 lam_min(QQ^T)^{1/2}, "
            "m_3 lam_min(QQ^T)^{1/4} ||(QQ^T)^{-1}Q||^{1/2}}) (eq. 4), where "
            "||C|| := max_i ||C_i|| and {m_i >= 0} depend on the problem's parameters, "
            "then there exists c_1 > 1 such that L(x^k,z^k,w^k) - L* in O(c_1^{-k}). "
            "Furthermore (x*,z*) is a local minimum of problem (1).*"),
        quantifiers=[
            "Equation (4) is stated with **unnamed constants** `{m_i}` inside a "
            "big-O, so it cannot be evaluated as literally printed. Appendix D "
            "(lines 5389-5560 and Theorem D.1 at line 5889) supplies its content: "
            "eq. (4) exists to force the reduced Hessian "
            "`nabla^2 f(x*) + sum_i w*_i C_i > 0`, which makes L Morse-Bott and gives "
            "the alpha=2 PL inequality behind the geometric rate.",
            "Two instantiations are therefore reported: an **a-priori certificate** "
            "built from the appendix's own bound (eqs. 29-31) on `||sum_i w*_i C_i||` "
            "using only problem data, and the **posterior operative condition** "
            "`lam_min(nabla^2 f(x*) + sum_i w*_i C_i) > 0`.",
            "The paper states explicitly (line 885) that `||C|| = 0` (linear "
            "constraints) satisfies eq. (4); the certificate must accept that case, "
            "and this is checked.",
            "A second conclusion, local minimality of `(x*,z*)`, is tested separately "
            "and directly.",
        ],
        measured=[
            "The a-priori certificate uses only `x^0`, `||C||`, `||d||`, `||e||`, `Q`, "
            "`L_f`, `mu_f`, `L_phi`, `mu_phi` - never the observed convergence rate. "
            "It is therefore not circular.",
            "`||C||` is swept geometrically on fixed problem geometry and the "
            "**empirical onset** of geometric convergence is located by bisection, "
            "independently of the certificate. A sufficient condition must be "
            "conservative: certificate threshold <= empirical onset. A certificate "
            "larger than the onset would contradict the theorem.",
            "Local minimality is tested **directly**, not by restating a second-order "
            "condition. Because Q has full row rank the feasible set projects onto "
            "{x : x_i in X_i} freely, so problem (1) reduces to "
            "`V(x) = f(x) + sum_i I_i(x_i) + min{phi(z) : Qz = -A(x)}`, whose inner "
            "minimisation has a closed form. Feasible points are then sampled in "
            "shells around x* and searched for any strict decrease.",
        ],
        control=(
            "Instances with `||C||` an order of magnitude above the bisected onset, "
            "where the reduced Hessian is indefinite. Geometric convergence must fail "
            "there. If it did not, the sweep would not be measuring what it claims."),
        raw=["raw/claim2/claim2_results.csv", "raw/claim2/claim2_bisection.json",
             "raw/claim2/claim2_localmin.json"],
        code=["repro/claims/claim2.py", "repro/analysis.py", "repro/localmin.py"],
        table_cols=["label", "norm_C", "eq4_margin", "eq4_certificate_holds",
                    "lam_min_reduced_hessian", "L_second_order_differentiable",
                    "determined", "linear", "c1_estimate", "is_local_minimum"],
        limitations=[
            "The constants `{m_i}` are not published, so the *literal* inequality of "
            "equation (4) cannot be evaluated. What is evaluated is the sufficient "
            "condition the appendix derives it from. This is an interpretation of the "
            "paper, stated openly; a reader who disagrees with it should read the "
            "posterior reduced-Hessian column instead, which is assumption-free.",
            "Local minimality is established by sampling, not by exhaustive search: it "
            "can refute local minimality by finding a better feasible point, but "
            "finding none is evidence rather than proof.",
        ],
    ),
    "claim3": dict(
        title="Claim 3 - Theorem 3.3 (polyhedral indicators)",
        slug="claim-3",
        statement=(
            "Theorem 3.3 (source line 896). *Under the assumptions of Theorem 3.1, "
            "when matrix Q satisfies equation 4 and {I_i} are the indicator functions "
            "of some polyhedral, then the iterates of ADMM satisfy "
            "L(x^k,z^k,w^k) - min_{(x,z) in B(x^k,z^k;r)} L(x,z,w^k) in O(c_2^{-k}), "
            "where c_2 > 1 and r > 0 are constant. Furthermore lim_k (x^k,z^k) = "
            "(x*,z*) is a local minimum of problem (1).*"),
        quantifiers=[
            "Theorem 3.3 exists to **drop** the second-order differentiability of L at "
            "the limit point that Theorem 3.2 assumes. L is second-order "
            "differentiable there whenever no indicator is active, so an instance "
            "whose iterates stay in the interior of its box tests Theorem 3.2's "
            "setting, not this one. Every instance here is checked to have at least "
            "one **active** constraint at the limit and to have L verifiably **not** "
            "second-order differentiable there.",
            "The decaying quantity is the **local gap** against the best point in a "
            "ball of fixed radius r with the dual frozen at w^k - not `L^k - L*`.",
        ],
        measured=[
            "`repro/localgap.py` computes the theorem's quantity explicitly: the inner "
            "problem `min L(x,z,w^k)` subject to `||(x,z)-(x^k,z^k)|| <= r` and "
            "`x_i in X_i` is solved by SLSQP from several starts. Finding a lower "
            "inner minimum makes the gap **larger**, so multi-start is the "
            "conservative direction and the reported gap is a lower bound on the "
            "theorem's quantity.",
            "The instances are the paper's own locomotion problem (friction pyramids "
            "plus the support band bind at the optimum by construction), the "
            "Section-5 toy with a box whose lower face binds, and a randomised "
            "ensemble with a binding box.",
        ],
        control=(
            "The same binding polyhedra with `||C||` pushed far outside the equation-(4) "
            "regime, which violates a hypothesis Theorem 3.3 shares with Theorem 3.2. "
            "The local gap must then fail to decay geometrically. A second, "
            "**descriptive** comparison replaces the polyhedra with Euclidean balls "
            "that are also active at the limit; Theorem 3.3 says nothing there, so it "
            "is reported and not scored."),
        raw=["raw/claim3/claim3_results.csv", "raw/claim3/claim3_local_gaps.json",
             "raw/claim3/claim3_ball_comparison.json"],
        code=["repro/claims/claim3.py", "repro/localgap.py", "repro/localmin.py"],
        table_cols=["label", "all_sets_polyhedral", "n_active_constraints",
                    "n_constraints", "min_slack", "L_second_order_differentiable",
                    "eq4_operative_condition", "determined", "local_gap_geometric",
                    "c2_estimate", "decades_of_decay", "is_local_minimum"],
        limitations=[
            "The inner minimisation is numerical, so the local gap has a solver "
            "accuracy floor (around 1e-10); decay is measured above that floor only.",
            "As for Claim 2, regime membership uses the operative form of eq. (4). "
            "The count of instances that also pass the stricter a-priori certificate "
            "is reported separately.",
        ],
    ),
    "claim4": dict(
        title="Claim 4 - Corollary 4.2 (the (dt)^3 term and t_0)",
        slug="claim-4",
        statement=(
            "Corollary 4.2 (source line 1332). *Under the assumptions of Theorem 4.1, "
            "if equation 7 holds and L(x,z,w) is second-order differentiable at the "
            "limit point, then there exist c_3 > 1 and t_0 > 0 such that the iterates "
            "of the ADMM applied to the problem in equation 6 with Delta t <= t_0 "
            "satisfy L(x^k,z^k,w^k) - L* in O(c_3^{-k}). Furthermore (x*,z*) is a "
            "local minimum of problem 6.* Supporting statement (line 1329): "
            "*equation 6 is a multi-affine quadratic constrained problem with the "
            "nonlinear term proportional to (Delta t)^3.*"),
        quantifiers=[
            "The `(Delta t)^3` statement is about the **constraint operator of eq. (6)**, "
            "so it is measured from the matrices actually assembled from the "
            "Newton-Euler dynamics: `||C(dt)|| = max_j ||C_j(dt)||_2`.",
            "`t_0` is an existential threshold; it is located by **bisection** on "
            "Delta t rather than read off a coarse grid.",
            "Equation (7) (`||x^0||^2, ||x*_f||^2 in O(n_x)`, `||z*_phi||^2 in O(n_z)`) "
            "is audited numerically on every configuration.",
        ],
        measured=[
            "`||C(dt)||` and `||d(dt)||` are read off the assembled operator over a "
            "geometric grid in Delta t and the exponent is fitted.",
            "ADMM is then run at every Delta t (with and without the polyhedral "
            "friction indicators) and classified.",
        ],
        control=(
            "The previous reproduction verified the `(Delta t)^3` scaling by "
            "regressing `log(dt**3)` on `log(dt)`. That is an identity: it returns "
            "3.000 for any input and carries no information about the paper. It is "
            "computed here **side by side** with the measured exponent, and flagged as "
            "vacuous, as an explicit control on the methodology itself."),
        raw=["raw/claim4/claim4_C_scaling.json", "raw/claim4/claim4_dt_sweep.csv",
             "raw/claim4/claim4_t0_bisection.json"],
        code=["repro/claims/claim4.py", "repro/problems/locomotion.py"],
        table_cols=["label", "dt", "norm_C", "norm_d", "determined", "linear",
                    "c3_estimate", "lam_min_reduced_hessian", "n_active_indicators",
                    "x0_sq_over_nx", "is_local_minimum"],
        limitations=[
            "`t_0` is bounded by bisection on one problem geometry; the theorem's "
            "`t_0` is instance-dependent, so the measured value is a property of the "
            "tested instance, not a universal constant.",
        ],
    ),
    "claim5": dict(
        title="Claim 5 - Figures 2 and 4 (q >= 10 and the baselines)",
        slug="claim-5",
        statement=(
            "Section 5 / Figure 2 (source line 1425): *Condition in (4) suggests "
            "q >= 10 to ensure linear convergence*, for the toy problem "
            "`min mu_x/2 (x1^2+x2^2+x3^2+x4^2) + mu_z/2 z^2 s.t. x1x2 - x3x4 + qz + 1 = 0`. "
            "Figure 4 / Appendix B (line 1437): *our algorithm achieves superior "
            "performance when the constraints are nonlinear, while comparable "
            "performance in other settings*, against PADMM (Yashtini 2021), IPDS-ADMM "
            "(Yuan 2025) and IADMM (Tang & Toh 2024)."),
        quantifiers=[
            "The q claim is one of **sufficiency**: every `q >= 10` converges linearly. "
            "It is **not** a claim that `q < 10` fails, so an empirical onset below 10 "
            "does **not** contradict it. The previous reproduction treated an observed "
            "transition at q=3 as confirmation of a q>=10 threshold; here sufficiency "
            "is tested over `q` up to 1000 across four `(mu_x, mu_z)` settings, three "
            "initial points and two `rho` multipliers, and the empirical onset is "
            "separately located by bisection.",
            "`mu_x` and `mu_z` are **never stated in the paper**, so they are swept "
            "rather than chosen.",
        ],
        measured=[
            "Geometric convergence across the full q grid, with any `q >= 10` failure "
            "reported explicitly as a counterexample to sufficiency.",
            "Iterations for `||A(x)+Qz||` to reach 1e-8 for each of the four "
            "algorithms on each of the three Appendix-B problems, over a sweep of "
            "`mu_x`, `mu_z`, `rho` and initial point. The final objective of each "
            "method's limit is reported alongside, because speed to feasibility and "
            "quality of the limit are different things.",
        ],
        control=(
            "The q sweep includes `q < 10`, where the paper's sufficient condition "
            "does not apply, so that the measured onset is separated from the claimed "
            "sufficient threshold rather than conflated with it."),
        raw=["raw/claim5/claim5_q_sweep.csv",
             "raw/claim5/claim5_q_onset_bisection.json",
             "raw/claim5/claim5_baseline_comparison.csv"],
        code=["repro/claims/claim5.py", "repro/problems/baselines.py"],
        table_cols=["problem", "mu_x", "mu_z", "rho", "winner",
                    "admm_iters_to_tol", "padmm_iters_to_tol", "iadmm_iters_to_tol",
                    "ipds_admm_iters_to_tol", "admm_final_obj", "admm_obj_is_best"],
        limitations=[
            "**Material deviation.** The three baseline papers (Yashtini 2021, Yuan "
            "2025, Tang & Toh 2024) were not retrievable through the literature "
            "tooling available here. The baselines are therefore reimplementations of "
            "the *standard published form* of each named method, not transcriptions of "
            "the original pseudocode. The comparison is a faithful comparison of "
            "exact-block-minimisation ADMM against proximal and linearised ADMM "
            "variants on the paper's own problems - evidence for the mechanism the "
            "paper appeals to, but not a certified reproduction of those three "
            "algorithms. This caps the confidence of this claim below HIGH regardless "
            "of the measured outcome.",
            "The figures' y-axis quantity is not labelled unambiguously in the source "
            "text, so the constraint residual is used and the objective is reported "
            "separately rather than assumed equivalent.",
        ],
    ),
    "claim6": dict(
        title="Claim 6 - Figures 5, 3 and 6 (locomotion and hardware)",
        slug="claim-6",
        statement=(
            "Section 5 / Figures 5 and 6. The method is validated on a 2D locomotion "
            "simulation (Figure 5: `m = 2 kg`, `f(f) = 0.5 sum ||f_i||^2`, "
            "`phi(z) = 5 sum ||k'_i||^2`, polyhedral friction cones; centre panel "
            "sweeps Delta t, right panel sweeps the initial configuration) **and** on "
            "real robot experiments including a humanoid vertical jump and a quadruped "
            "bounding task (Figure 6), with Figure 3 reporting the mean and standard "
            "deviation of the centroidal dynamics constraint violation per ADMM "
            "iteration over 10 trials with randomised initial conditions, for three "
            "Delta t."),
        quantifiers=[
            "The claim is a **conjunction**: simulation *and* real robot experiments. "
            "Both conjuncts must hold for the claim to hold.",
            "The simulation conjunct, and the *numerical* content of the robot "
            "experiments (the centroidal trajectory optimisation of eq. (5)/(6) solved "
            "by Algorithm 1, including the Figure-3 violation curves), are fully "
            "reproducible and are reproduced here with complete contact schedules "
            "including flight phases.",
            "The **hardware execution** - trajectories tracked by a DDP kinematics "
            "optimiser and run on the physical robots - is not reproducible without "
            "those robots, and the authors' hardware logs are not published.",
        ],
        measured=[
            "Figure 5 centre: Delta t sweep on the 2D problem at the paper's parameters.",
            "Figure 5 right: 10 randomised initial configurations.",
            "Figure 3: humanoid vertical jump (3D, 55 kg, stance-flight-landing "
            "schedule) at three Delta t, 10 randomised initial conditions each; the "
            "reported quantity is exactly the paper's - mean and standard deviation of "
            "`||A(x)+Qz||` per ADMM iteration.",
            "Figure 6: the centroidal solves behind both motions - humanoid jump and "
            "quadruped bounding gait (4 contacts, alternating pairs with flight "
            "phases).",
        ],
        control=(
            "Every run is checked to start from a strictly feasible force trajectory "
            "and to end with the sub-problem KKT residual at solver tolerance, so a "
            "'converged' result cannot come from an infeasible or unsolved "
            "sub-problem."),
        raw=["raw/claim6/claim6_results.csv",
             "raw/claim6/claim6_fig3_violation_mean_std.json",
             "raw/claim6/claim6_violation_traces.json"],
        code=["repro/claims/claim6.py", "repro/problems/locomotion.py"],
        table_cols=["label", "dt", "gait", "n_x", "n_c", "norm_C",
                    "n_active_indicators", "determined", "geometric",
                    "decades_of_decay", "final_violation", "subproblem_max_kkt"],
        limitations=[
            "**The hardware conjunct cannot be verified or falsified here.** No robot "
            "hardware is available and the authors' hardware trajectory and tracking "
            "logs are not published. Because the claim is a conjunction, the honest "
            "overall verdict is BLOCKED even though the simulation half reproduces "
            "cleanly. It is not reported as a pass.",
            "What would unblock it: access to the two robots (or the authors' released "
            "hardware logs) together with the Crocoddyl DDP tracking configuration "
            "used in the paper.",
            "Section 5 says the 2D constraints 'are designed to ensure that the center "
            "of mass remains within a specified target area'. A CoM box couples force "
            "blocks and is therefore **not** block-separable, which Assumption 2.3 "
            "requires. It is imposed here in the separable form the assumption allows: "
            "at every time step the total normal force stays within a band around body "
            "weight. Without such a term the least-norm objective is minimised by "
            "f = 0 (free fall) at the friction-cone apex and the instance is "
            "degenerate. This deviation is a choice made by this reproduction.",
        ],
    ),
}
