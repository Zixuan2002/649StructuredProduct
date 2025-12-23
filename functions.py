import numpy as np

def option_pricing_fd(s1, s2, s3, K, T, bar1, bar2, c, N, r, S0):
    """
    s1,s2,s3: each shape (M, steps+1) paths for TSLA/MSFT/META (or whatever order you use)
    K: strike ratio (e.g. 0.80)
    bar1, bar2: KO barriers for phase1/phase2 (e.g. 1.20, 1.10)
    c: coupon rate p.a. (e.g. 0.10)
    N: notional (e.g. 1e6)
    r: risk-free rate
    Returns: array of PVs, shape (M,)
    """
    M = s1.shape[0]
    payoffs = np.zeros(M)

    for j in range(M):
        path1 = s1[j]
        path2 = s2[j]
        path3 = s3[j]

        # Phase 1: indices 0..251 (1 year)
        p0 = np.minimum.reduce([
            path1[0:252] / S0[0],
            path2[0:252] / S0[1],
            path3[0:252] / S0[2]
        ])

        # Phase 2: indices 252..end (reset at 252)
        p1 = np.minimum.reduce([
            path1[252:] / path1[252],
            path2[252:] / path2[252],
            path3[252:] / path3[252]
        ])

        awarded = False

        # -------- Phase 1 KO: redeem immediately on day i --------
        for i in range(1, len(p0)):
            if p0[i] >= bar1:
                tau = i / 252.0  # KO time in years (immediate)
                pv = 0.0

                # coupons paid on scheduled quarter dates <= tau
                # (no stub; quarter dates are 0.25,0.5,0.75,1.0,...)
                for l in range(1, 9):
                    t_coupon = l / 4.0
                    if t_coupon <= tau:
                        pv += np.exp(-r * t_coupon) * (c / 4.0) * N

                # redemption at KO time tau
                pv += np.exp(-r * tau) * N
                payoffs[j] = pv
                awarded = True
                break

        # -------- Phase 2 KO: redeem immediately on day 1y + i --------
        if not awarded:
            for i in range(1, len(p1)):
                if p1[i] >= bar2:
                    tau = 1.0 + i / 252.0
                    pv = 0.0
                    for l in range(1, 9):
                        t_coupon = l / 4.0
                        if t_coupon <= tau:
                            pv += np.exp(-r * t_coupon) * (c / 4.0) * N
                    pv += np.exp(-r * tau) * N
                    payoffs[j] = pv
                    awarded = True
                    break

        # -------- No KO: maturity payoff --------
        if not awarded:
            pv = 0.0
            # all 8 coupons
            for l in range(1, 9):
                pv += np.exp(-r * (l / 4.0)) * (c / 4.0) * N

            P_T = p1[-1]  # phase-2 worst-of performance at maturity (for the K test)

            if P_T >= K:
                pv += np.exp(-r * T) * N
            else:
                # exact "worst stock" settlement relative to ISSUE DATE initial
                ratios_to_issue = np.array([
                    path1[-1] / path1[0],
                    path2[-1] / path2[0],
                    path3[-1] / path3[0]
                ])
                j_star = int(np.argmin(ratios_to_issue))

                S0_star = np.array([path1[0], path2[0], path3[0]])[j_star]
                ST_star = np.array([path1[-1], path2[-1], path3[-1]])[j_star]

                shares = N / (K * bar1 * S0_star)
                pv += np.exp(-r * T) * shares * ST_star

            payoffs[j] = pv

    return payoffs
