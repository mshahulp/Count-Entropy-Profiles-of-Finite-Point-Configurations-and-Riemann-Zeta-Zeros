
import argparse
import sys

# ============================================================
# FIGURE 1
# ============================================================
def generate_figure1():


    import argparse
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np
    import pandas as pd


    DISPLAY_LIMIT = 20.0
    LOG_FLOOR = 1e-6


    def parse_args():
        p = argparse.ArgumentParser(description="Generate final manuscript Figure 1.")
        p.add_argument("--gue", type=Path,
                       default=Path("gue_master_profile_summary.csv"))
        p.add_argument("--poisson", type=Path,
                       default=Path("poisson_master_profile_summary.csv"))
        p.add_argument("--lattice", type=Path,
                       default=Path("lattice_entropy_profile.csv"))
        p.add_argument("--pmf", type=Path,
                       default=Path("reference_mean_count_pmfs.csv"))
        p.add_argument("--output-dir", type=Path,
                       default=Path("figure1_final_output"))
        p.add_argument("--basename",
                       default="Figure1_reference_count_entropy")
        return p.parse_args([])


    def require_columns(df, cols, name):
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")


    def prepare_pmf(pmf):
        d = pmf[pmf["source"].isin(["GUE", "Poisson"])].copy()
        if d.empty:
            raise ValueError("PMF file contains no GUE or Poisson rows.")

        d["kp"] = d["k"] * d["p"]

        means = (
            d.groupby(["source", "L"], as_index=False)
             .agg(mean_count=("kp", "sum"),
                  pmf_sum=("p", "sum"))
        )

        d = d.merge(means, on=["source", "L"], how="left",
                    validate="many_to_one")
        d["k_minus_mean"] = d["k"] - d["mean_count"]

        return d


    def display_range_qc(d):
        rows = []

        for (source, L), g in d.groupby(["source", "L"]):
            outside = g.loc[
                np.abs(g["k_minus_mean"]) > DISPLAY_LIMIT, "p"
            ].sum()

            below_floor_inside = g.loc[
                (np.abs(g["k_minus_mean"]) <= DISPLAY_LIMIT)
                & (g["p"] < LOG_FLOOR),
                "p"
            ].sum()

            rows.append({
                "source": source,
                "L": L,
                "pmf_sum": g["p"].sum(),
                "mass_outside_display_range": outside,
                "mass_below_log_floor_inside_range": below_floor_inside,
            })

        return pd.DataFrame(rows)


    def main():
        args = parse_args()
        args.output_dir.mkdir(parents=True, exist_ok=True)

        gue = pd.read_csv(args.gue)
        poisson = pd.read_csv(args.poisson)
        lattice = pd.read_csv(args.lattice)
        pmf = pd.read_csv(args.pmf)

        require_columns(gue, ["L", "mean", "q025", "q975"], "GUE summary")
        require_columns(poisson, ["L", "mean", "q025", "q975"],
                        "Poisson summary")
        require_columns(lattice, ["L", "H_lattice"], "Lattice profile")
        require_columns(pmf, ["source", "L", "k", "p"], "Reference PMF")

        d = prepare_pmf(pmf)

        qc = display_range_qc(d)
        qc_path = args.output_dir / "Figure1_display_range_qc.csv"
        qc.to_csv(qc_path, index=False)

        # Common log normalization across GUE and Poisson.
        p_max = float(d["p"].max())
        norm = LogNorm(vmin=LOG_FLOOR, vmax=p_max)

        fig = plt.figure(figsize=(9.0, 11.0))
        gs = fig.add_gridspec(
            3, 1,
            height_ratios=[1.15, 1.0, 1.0],
            hspace=0.32
        )

        # --------------------------------------------------------
        # (a) Reference entropy profiles
        # --------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0])

        ax1.plot(
            lattice["L"], lattice["H_lattice"],
            linewidth=1.8, label="Lattice"
        )

        ax1.plot(
            gue["L"], gue["mean"],
            linewidth=1.8, label="GUE"
        )
        ax1.fill_between(
            gue["L"], gue["q025"], gue["q975"],
            alpha=0.18
        )

        ax1.plot(
            poisson["L"], poisson["mean"],
            linewidth=1.8, label="Poisson"
        )
        ax1.fill_between(
            poisson["L"], poisson["q025"], poisson["q975"],
            alpha=0.18
        )

        ax1.set_xlim(0.1, 20.0)
        ax1.set_xlabel(r"Window length $L$")
        ax1.set_ylabel(r"Count entropy $H(L)$ (nats)")
        ax1.set_title("(a) Reference count-entropy profiles")
        ax1.legend(frameon=False, ncol=3)

        # --------------------------------------------------------
        # (b,c) Ensemble-mean count laws
        # --------------------------------------------------------
        pmf_axes = []
        handle = None

        for row, source, title in [
            (1, "GUE", "(b) GUE ensemble-mean count law"),
            (2, "Poisson", "(c) Poisson ensemble-mean count law"),
        ]:
            ax = fig.add_subplot(gs[row, 0])
            pmf_axes.append(ax)

            panel = d[
                (d["source"] == source)
                & (np.abs(d["k_minus_mean"]) <= DISPLAY_LIMIT)
                & (d["p"] >= LOG_FLOOR)
            ]

            handle = ax.scatter(
                panel["L"],
                panel["k_minus_mean"],
                c=panel["p"],
                s=18,
                marker="s",
                norm=norm,
            )

            ax.axhline(0.0, linewidth=0.8, alpha=0.6)
            ax.set_xlim(0.1, 20.0)
            ax.set_ylim(-DISPLAY_LIMIT, DISPLAY_LIMIT)
            ax.set_xlabel(r"Window length $L$")
            ax.set_ylabel(r"Centered count $k-\mathbb{E}[K_L]$")
            ax.set_title(title)

        cbar = fig.colorbar(
            handle,
            ax=pmf_axes,
            fraction=0.025,
            pad=0.025,
        )
        cbar.set_label(r"Probability mass $p_L(k)$ (log scale)")

        fig.align_ylabels()

        svg = args.output_dir / f"{args.basename}.svg"
        pdf = args.output_dir / f"{args.basename}.pdf"
        png = args.output_dir / f"{args.basename}.png"

        fig.savefig(svg, format="svg", bbox_inches="tight")
        fig.savefig(pdf, format="pdf", bbox_inches="tight")
        fig.savefig(png, format="png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        # Console QC summary.
        print("\nFigure 1 generated successfully.")
        print(f"SVG: {svg.resolve()}")
        print(f"PDF: {pdf.resolve()}")
        print(f"PNG: {png.resolve()}")
        print(f"QC : {qc_path.resolve()}")

        print("\nDisplay conventions:")
        print(f"  centered-count range = [-{DISPLAY_LIMIT:g}, {DISPLAY_LIMIT:g}]")
        print(f"  logarithmic probability floor = {LOG_FLOOR:g}")

        for source in ["GUE", "Poisson"]:
            q = qc[qc["source"] == source]
            print(
                f"\n{source}:"
                f"\n  maximum mass outside displayed range = "
                f"{q['mass_outside_display_range'].max():.10g}"
                f"\n  maximum mass below log floor inside range = "
                f"{q['mass_below_log_floor_inside_range'].max():.10g}"
                f"\n  maximum PMF normalization error = "
                f"{np.max(np.abs(q['pmf_sum'] - 1.0)):.3e}"
            )


    if __name__ == "__main__":
        main()


# ============================================================
# FIGURE 2
# ============================================================
def generate_figure2():

    import argparse
    from pathlib import Path
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    REGIME_ORDER = ["low", "midlow", "1e12", "1e21", "1e22"]
    DISPLAY = {
        "low": r"First $10^4$ zeros",
        "midlow": r"Zeros $45{,}001$--$55{,}000$",
        "1e12": r"Near $10^{12}$",
        "1e21": r"Near $10^{21}$",
        "1e22": r"Near $10^{22}$",
    }

    def args():
        p = argparse.ArgumentParser()
        p.add_argument("--profiles", type=Path,
                       default=Path("zeta_count_entropy_profiles.csv"))
        p.add_argument("--output-dir", type=Path,
                       default=Path("figure2_final_output"))
        return p.parse_args([])

    def first_existing(cols, candidates):
        m = {str(c).lower(): c for c in cols}
        for x in candidates:
            if x.lower() in m:
                return m[x.lower()]
        return None

    def canon(v):
        s = str(v).strip().lower()
        c = re.sub(r"[\s_\-]+", "", s)
        if "1e22" in c or "10^22" in c or "10²²" in c: return "1e22"
        if "1e21" in c or "10^21" in c or "10²¹" in c: return "1e21"
        if "1e12" in c or "10^12" in c or "10¹²" in c: return "1e12"
        if "midlow" in c or "45001" in c or "robust" in c: return "midlow"
        if "low" in c or c in {"first", "first10000", "first10k"}: return "low"
        return s

    def load_profiles(path):
        x = pd.read_csv(path)
        L = first_existing(x.columns, ["L","window_length","window","scale"])
        H = first_existing(x.columns, ["H","entropy","count_entropy","H_L"])
        R = first_existing(x.columns, ["regime","source_regime","height_regime","dataset","source"])
        B = first_existing(x.columns, ["block","block_id","configuration","config","config_id","label"])
        if any(v is None for v in [L,H,R,B]):
            raise ValueError(f"Could not infer required columns. Available: {list(x.columns)}")
        d = x[[L,H,R,B]].copy()
        d.columns = ["L","H","regime_raw","block"]
        d["regime"] = d["regime_raw"].map(canon)
        unknown = sorted(set(d["regime"]) - set(REGIME_ORDER))
        if unknown:
            raise ValueError(f"Unrecognized regimes: {unknown}")
        d["L"] = pd.to_numeric(d["L"])
        d["H"] = pd.to_numeric(d["H"])
        return d

    def main():
        a = args()
        a.output_dir.mkdir(parents=True, exist_ok=True)
        d = load_profiles(a.profiles)

        means = (d.groupby(["regime","L"], as_index=False)
                   .agg(H_mean=("H","mean"),
                        H_min=("H","min"),
                        H_max=("H","max"),
                        H_sd=("H","std"),
                        n_blocks=("block","nunique")))
        P = means.pivot(index="regime", columns="L", values="H_mean").reindex(REGIME_ORDER)
        if P.isna().any().any():
            raise ValueError("Incomplete common L grid across regimes.")

        # Consecutive displacement table.
        pairs = list(zip(REGIME_ORDER[:-1], REGIME_ORDER[1:]))
        C = pd.DataFrame(
            [P.loc[b] - P.loc[a] for a,b in pairs],
            index=[f"{b} - {a}" for a,b in pairs]
        )
        C.columns = P.columns

        # Pairwise RMS distances between complete mean profiles.
        arr = P.to_numpy(float)
        D = np.zeros((len(REGIME_ORDER), len(REGIME_ORDER)))
        for i in range(len(REGIME_ORDER)):
            for j in range(len(REGIME_ORDER)):
                D[i,j] = np.sqrt(np.mean((arr[i]-arr[j])**2))
        D = pd.DataFrame(D, index=REGIME_ORDER, columns=REGIME_ORDER)

        means.to_csv(a.output_dir/"Figure2_regime_mean_profiles_final.csv", index=False)
        C.to_csv(a.output_dir/"Figure2_consecutive_regime_differences_final.csv")
        D.to_csv(a.output_dir/"Figure2_pairwise_RMS_profile_distances_final.csv")

        # ------------------ publication figure ------------------
        fig = plt.figure(figsize=(11.2, 8.4))
        gs = fig.add_gridspec(
            2, 2, height_ratios=[1.15, 1.0],
            hspace=0.34, wspace=0.32
        )

        # (a) complete profiles
        ax = fig.add_subplot(gs[0,:])
        for regime in REGIME_ORDER:
            rd = d[d.regime == regime]
            for _, bd in rd.groupby("block"):
                bd = bd.sort_values("L")
                ax.plot(bd.L, bd.H, lw=0.65, alpha=0.16)
            rm = means[means.regime == regime].sort_values("L")
            ax.plot(rm.L, rm.H_mean, lw=2.0, label=DISPLAY[regime])
        ax.set_xlim(float(d.L.min()), float(d.L.max()))
        ax.set_xlabel(r"Window length $L$")
        ax.set_ylabel(r"Count entropy $H(L)$ (nats)")
        ax.set_title("(a) Count-entropy profiles across spectral height", loc="left")
        ax.legend(frameon=False, ncol=3, fontsize=8.5)

        # (b) scale-resolved consecutive changes
        ax = fig.add_subplot(gs[1,0])
        vals = C.to_numpy(float)
        vmax = float(np.max(np.abs(vals)))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.imshow(
            vals, aspect="auto", origin="upper",
            extent=[float(C.columns.min()), float(C.columns.max()),
                    len(C.index)-0.5, -0.5],
            norm=norm, interpolation="nearest"
        )
        ax.set_yticks(np.arange(4))
        ax.set_yticklabels([
            r"45,001--55,000 $-$ first $10^4$",
            r"$10^{12}$ $-$ 45,001--55,000",
            r"$10^{21}$ $-$ $10^{12}$",
            r"$10^{22}$ $-$ $10^{21}$",
        ], fontsize=8)
        ax.set_xlabel(r"Window length $L$")
        ax.set_title("(b) Consecutive-regime entropy displacement", loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.04)
        cb.set_label("Entropy difference")

        # (c) global functional distances
        ax = fig.add_subplot(gs[1,1])
        M = D.to_numpy(float)
        im2 = ax.imshow(M, origin="upper", aspect="equal",
                        interpolation="nearest")
        labels = [
            "First\n$10^4$",
            "45,001--\n55,000",
            "$10^{12}$",
            "$10^{21}$",
            "$10^{22}$",
        ]
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks(np.arange(5))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title("(c) Pairwise RMS profile distance", loc="left")
        threshold = (M.max() + M.min())/2
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{M[i,j]:.4f}",
                        ha="center", va="center", fontsize=7.5)
        cb2 = fig.colorbar(im2, ax=ax, fraction=0.047, pad=0.04)
        cb2.set_label("RMS profile distance")

        stem = a.output_dir/"Figure2_zeta_profiles_across_height"
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
        plt.close(fig)

        print("\nPairwise RMS distances:")
        print(D.round(6).to_string())
        print("\nOutputs written to:", a.output_dir.resolve())

    if __name__ == "__main__":
        main()


# ============================================================
# FIGURE 3
# ============================================================
def generate_figure3():
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    HERE = Path.cwd()

    def find_input(stem):
        candidates = [HERE / f"{stem}.csv", HERE / f"{stem}(2).csv", HERE / f"{stem} (2).csv"]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"Could not find {stem}.csv in:\n{HERE}\nPlace the two required CSV files in the same folder as this script.")

    GUE_FILE = find_input("gue_self_distance_distribution")
    ZETA_FILE = find_input("zeta_gue_block_distances")
    gue = pd.read_csv(GUE_FILE)
    zeta = pd.read_csv(ZETA_FILE)
    print("Reading:")
    print(" ", GUE_FILE.name)
    print(" ", ZETA_FILE.name)

    gue_required = {"realization", "rmse_to_leave_one_out_GUE_mean", "sup_to_leave_one_out_GUE_mean"}
    zeta_required = {"role", "regime", "block", "rmse_to_GUE_mean", "sup_to_GUE_mean"}
    missing_gue = gue_required - set(gue.columns)
    missing_zeta = zeta_required - set(zeta.columns)
    if missing_gue: raise ValueError(f"GUE CSV is missing columns: {sorted(missing_gue)}")
    if missing_zeta: raise ValueError(f"Zeta CSV is missing columns: {sorted(missing_zeta)}")
    if len(gue) != 400: raise ValueError(f"Expected 400 GUE realizations; found {len(gue)}.")
    if len(zeta) != 20: raise ValueError(f"Expected 20 zeta blocks; found {len(zeta)}.")

    def normalize_regime(x):
        s = str(x).strip().lower().replace(" ", "")
        aliases = {"low":"low", "first1e4":"low", "first10^4":"low", "first10000":"low", "midlow":"midlow", "midlow_45k_55k":"midlow", "45001-55000":"midlow", "45,001-55,000":"midlow", "1e12":"1e12", "10^12":"1e12", "1e21":"1e21", "10^21":"1e21", "1e22":"1e22", "10^22":"1e22"}
        return aliases.get(s, s)

    zeta["regime_key"] = zeta["regime"].map(normalize_regime)
    expected = ["low", "midlow", "1e12", "1e21", "1e22"]
    found = set(zeta["regime_key"])
    if set(expected) != found:
        raise ValueError(f"Unexpected regime labels.\nFound: {sorted(found)}\nExpected: {expected}\nInspect zeta['regime'] if your labels differ.")
    counts = zeta.groupby("regime_key").size()
    for key in expected:
        if counts[key] != 4: raise ValueError(f"Expected 4 zeta blocks for {key}; found {counts[key]}.")

    D_gue = gue["rmse_to_leave_one_out_GUE_mean"].to_numpy(float)
    Delta_gue = gue["sup_to_leave_one_out_GUE_mean"].to_numpy(float)
    zeta["D"] = zeta["rmse_to_GUE_mean"].astype(float)
    zeta["Delta"] = zeta["sup_to_GUE_mean"].astype(float)
    q50, q90, q95, q975, q99 = np.quantile(D_gue, [0.50,0.90,0.95,0.975,0.99])
    print("\nValidation")
    for label, value in [("GUE realizations",len(gue)),("Zeta blocks",len(zeta))]: print(f"  {label}: {value}")
    for label, value in [("D_GUE 50th",q50),("D_GUE 90th",q90),("D_GUE 95th",q95),("D_GUE 97.5th",q975),("D_GUE 99th",q99)]: print(f"  {label}: {value:.5f}")

    styles = {
    "low":{"label":r"First $10^4$ zeros","short":r"First $10^4$","marker":"o","color":"tab:orange","size":55},
    "midlow":{"label":"Zeros 45,001–55,000","short":"45,001–55,000","marker":"s","color":"tab:green","size":55},
    "1e12":{"label":r"Near $10^{12}$","short":r"Near $10^{12}$","marker":"^","color":"tab:red","size":62},
    "1e21":{"label":r"Near $10^{21}$","short":r"Near $10^{21}$","marker":"D","color":"tab:purple","size":55},
    "1e22":{"label":r"Near $10^{22}$","short":r"Near $10^{22}$","marker":"*","color":"tab:cyan","size":85}}

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"axes.titlesize":10.5,"axes.labelsize":9.5,"xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":8.0,"axes.linewidth":0.8,"pdf.fonttype":42,"ps.fonttype":42})
    fig = plt.figure(figsize=(12.6,5.7))
    gs = fig.add_gridspec(2,2,width_ratios=[1.02,1.0],height_ratios=[3.5,1.15],left=0.10,right=0.985,bottom=0.19,top=0.94,wspace=0.20,hspace=0.03)
    ax_a=fig.add_subplot(gs[0,0]); ax_strip=fig.add_subplot(gs[1,0],sharex=ax_a); ax_b=fig.add_subplot(gs[:,1])

    D_sorted=np.sort(D_gue); ecdf=np.arange(1,len(D_sorted)+1)/len(D_sorted)
    ax_a.plot(D_sorted,ecdf,lw=1.5,color="tab:blue"); ax_a.axvline(q95,ls="--",lw=1.0,color="0.35"); ax_a.axvline(q99,ls=":",lw=1.0,color="0.35")
    ax_a.text(q95,0.77,f"95th\n{q95:.5f}",ha="right",va="top",fontsize=8.0); ax_a.text(q99,0.77,f"99th\n{q99:.5f}",ha="left",va="top",fontsize=8.0)
    ax_a.set_ylabel("Empirical cumulative probability"); ax_a.set_title("(a) GUE leave-one-out calibration",loc="left"); ax_a.set_ylim(0,1.02); ax_a.tick_params(axis="x",labelbottom=False)
    xmax=max(zeta["D"].max(),D_gue.max())*1.07; ax_a.set_xlim(0,xmax)
    ax_ai=inset_axes(ax_a,width="47%",height="49%",loc="lower right",borderpad=1.15); ax_ai.plot(D_sorted,ecdf,lw=1.1,color="tab:blue"); ax_ai.axvline(q95,ls="--",lw=0.8,color="0.35"); ax_ai.axvline(q99,ls=":",lw=0.8,color="0.35"); ax_ai.set_xlim(0,0.06); ax_ai.set_ylim(0,1.02); ax_ai.set_title(r"Zoom: $D\leq0.06$",fontsize=8.0); ax_ai.tick_params(labelsize=7.0)

    strip_order=["low","midlow","1e12","1e21","1e22"]
    for y,key in enumerate(strip_order):
        sub=zeta[zeta["regime_key"]==key]; st=styles[key]; ax_strip.axhline(y,color="0.85",lw=0.6,zorder=0); ax_strip.scatter(sub["D"],np.full(len(sub),y),marker=st["marker"],s=st["size"],color=st["color"],edgecolor="black",linewidth=0.45,zorder=3)
    ax_strip.set_yticks(range(len(strip_order))); ax_strip.set_yticklabels([styles[k]["short"] for k in strip_order]); ax_strip.set_ylim(-0.55,len(strip_order)-0.45); ax_strip.set_xlabel(r"RMS profile discrepancy, $D$"); ax_strip.spines["top"].set_visible(False)

    ax_b.scatter(D_gue,Delta_gue,s=16,marker="o",color="tab:blue",alpha=0.25,edgecolors="none",zorder=1)
    for key in expected:
        sub=zeta[zeta["regime_key"]==key]; st=styles[key]; ax_b.scatter(sub["D"],sub["Delta"],marker=st["marker"],s=st["size"],color=st["color"],edgecolor="black",linewidth=0.45,zorder=4)
    ax_b.set_xlabel(r"RMS profile discrepancy, $D$"); ax_b.set_ylabel(r"Maximum absolute profile discrepancy, $\Delta$"); ax_b.set_title("(b) Joint profile discrepancy",loc="left"); ax_b.set_xlim(0,xmax); ymax=max(zeta["Delta"].max(),Delta_gue.max())*1.08; ax_b.set_ylim(0,ymax)
    zoom_D=0.06; zoom_Delta=0.10; ax_b.add_patch(Rectangle((0,0),zoom_D,zoom_Delta,fill=False,ls="--",lw=0.9,edgecolor="0.45",zorder=2))
    ax_bi=inset_axes(ax_b,width="46%",height="46%",loc="upper left",borderpad=1.20); ax_bi.scatter(D_gue,Delta_gue,s=11,marker="o",color="tab:blue",alpha=0.25,edgecolors="none",zorder=1)
    for key in expected:
        sub=zeta[zeta["regime_key"]==key]; st=styles[key]; ax_bi.scatter(sub["D"],sub["Delta"],marker=st["marker"],s=st["size"]*0.65,color=st["color"],edgecolor="black",linewidth=0.35,zorder=4)
    ax_bi.set_xlim(0,zoom_D); ax_bi.set_ylim(0,zoom_Delta); ax_bi.set_title(r"Zoom: $D\leq0.06,\ \Delta\leq0.10$",fontsize=8.0); ax_bi.tick_params(labelsize=7.0)

    legend_handles=[Line2D([0],[0],marker="o",linestyle="none",markersize=5.5,markerfacecolor="tab:blue",markeredgecolor="none",alpha=0.40,label="GUE leave-one-out realizations")]
    for key in expected:
        st=styles[key]; legend_handles.append(Line2D([0],[0],marker=st["marker"],linestyle="none",markersize=7.0 if st["marker"]!="*" else 8.5,markerfacecolor=st["color"],markeredgecolor="black",markeredgewidth=0.45,label=st["label"]))
    fig.legend(handles=legend_handles,loc="lower center",bbox_to_anchor=(0.54,0.045),ncol=3,frameon=False,handletextpad=0.55,columnspacing=1.45)

    summary=(zeta.groupby("regime_key",sort=False).agg(n=("D","size"),D_mean=("D","mean"),D_min=("D","min"),D_max=("D","max"),Delta_mean=("Delta","mean")).reindex(expected))
    print("\nZeta regime summary reconstructed from plotted data:"); print(summary.to_string(float_format=lambda x:f"{x:.6f}"))
    expected_D_means={"low":0.27297,"midlow":0.21676,"1e12":0.07745,"1e21":0.03850,"1e22":0.03912}
    for key,target in expected_D_means.items():
        observed=summary.loc[key,"D_mean"]
        if not np.isclose(observed,target,atol=5e-5): raise ValueError(f"Validation failed for {key}: observed D mean={observed:.8f}, expected≈{target:.5f}")
    print("\nValidation passed: regime-level D means agree with the manuscript.")

    base="Figure3_GUE_comparison_corrected"
    fig.savefig(f"{base}.png",dpi=600,bbox_inches="tight"); fig.savefig(f"{base}.pdf",bbox_inches="tight"); fig.savefig(f"{base}.svg",bbox_inches="tight")
    plt.close(fig)
    print("\nSaved:"); print(f"  {base}.png"); print(f"  {base}.pdf"); print(f"  {base}.svg")


# ============================================================
# FIGURE 4
# ============================================================
def generate_figure4():

    from pathlib import Path
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec


    # ============================================================
    # Paths
    # ============================================================

    BASE = Path(".")
    BLOCK_FILE = BASE / "surrogate_block_comparison.csv"
    POINTWISE_FILE = BASE / "surrogate_pointwise_summary.csv"
    SELF_FILE = BASE / "surrogate_self_distance_distribution.csv"

    OUT = BASE / "figure4_final_output"
    OUT.mkdir(parents=True, exist_ok=True)


    # ============================================================
    # Load data
    # ============================================================

    bc = pd.read_csv(BLOCK_FILE)
    pw = pd.read_csv(POINTWISE_FILE)
    sd = pd.read_csv(SELF_FILE)

    required_bc = {
        "regime",
        "block",
        "H_rmse_original_to_surrogate_mean",
        "H_surrogate_LOO_rmse_percentile",
        "H_empirical_upper_tail",
        "H_mean_signed_difference",
    }

    required_pw = {
        "regime",
        "block",
        "L",
        "H_original_minus_surrogate_mean",
    }

    required_sd = {
        "regime",
        "block",
        "surrogate",
        "H_rmse_to_LOO_surrogate_mean",
    }

    for name, frame, cols in [
        ("block comparison", bc, required_bc),
        ("pointwise summary", pw, required_pw),
        ("surrogate self-distance", sd, required_sd),
    ]:
        missing = cols - set(frame.columns)
        if missing:
            raise ValueError(f"{name} file missing columns: {sorted(missing)}")


    # ============================================================
    # Regime normalization
    # ============================================================

    ORDER = ["low", "midlow", "1e12", "1e21", "1e22"]

    DISPLAY = {
        "low": r"First $10^4$ zeros",
        "midlow": r"$45{,}001$–$55{,}000$",
        "1e12": r"Near $10^{12}$",
        "1e21": r"Near $10^{21}$",
        "1e22": r"Near $10^{22}$",
    }

    SHORT = {
        "low": r"First $10^4$",
        "midlow": r"$45{,}001$–$55{,}000$",
        "1e12": r"$10^{12}$",
        "1e21": r"$10^{21}$",
        "1e22": r"$10^{22}$",
    }


    def canon(v):
        s = str(v).strip().lower()
        c = re.sub(r"[\s_\-–—]+", "", s)

        if "1e22" in c or "10^22" in c:
            return "1e22"
        if "1e21" in c or "10^21" in c:
            return "1e21"
        if "1e12" in c or "10^12" in c:
            return "1e12"
        if "midlow" in c or "45001" in c or "robust" in c:
            return "midlow"
        if "low" in c or c in {"first", "first10000", "first10k"}:
            return "low"

        return s


    for frame in (bc, pw, sd):
        frame["regime_plot"] = frame["regime"].map(canon)

    unknown = (
        set(bc["regime_plot"])
        | set(pw["regime_plot"])
        | set(sd["regime_plot"])
    ) - set(ORDER)

    if unknown:
        raise ValueError(f"Unrecognized regime labels: {sorted(unknown)}")


    # ============================================================
    # Panel (a): regime-level signed-displacement curves
    # ============================================================

    regime_curves = {}

    for regime in ORDER:
        sub = pw[pw["regime_plot"] == regime]

        pivot = sub.pivot(
            index="L",
            columns="block",
            values="H_original_minus_surrogate_mean",
        ).sort_index()

        regime_curves[regime] = {
            "L": pivot.index.to_numpy(float),
            "mean": pivot.mean(axis=1).to_numpy(float),
            "min": pivot.min(axis=1).to_numpy(float),
            "max": pivot.max(axis=1).to_numpy(float),
        }


    # ============================================================
    # Panel (b): complete configuration × scale matrix
    # ============================================================

    matrix_rows = []
    row_labels = []
    group_centres = []
    group_boundaries = []

    row_index = 0

    for regime_i, regime in enumerate(ORDER):
        blocks = sorted(
            pw.loc[pw["regime_plot"] == regime, "block"].unique()
        )

        start = row_index

        for block in blocks:
            row = (
                pw[
                    (pw["regime_plot"] == regime)
                    & (pw["block"] == block)
                ]
                .sort_values("L")
            )

            matrix_rows.append(
                row["H_original_minus_surrogate_mean"].to_numpy(float)
            )
            row_labels.append((regime, int(block)))
            row_index += 1

        end = row_index - 1
        group_centres.append((start + end) / 2)

        if regime_i < len(ORDER) - 1:
            group_boundaries.append(end + 0.5)

    disp = np.vstack(matrix_rows)
    L_grid = np.sort(pw["L"].unique().astype(float))

    negative_fraction = float(np.mean(disp < -1e-12))


    # ============================================================
    # Panel (c): paired empirical calibration
    # ============================================================

    max_loo = (
        sd.groupby(["regime_plot", "block"], as_index=False)
        ["H_rmse_to_LOO_surrogate_mean"]
        .max()
        .rename(
            columns={
                "H_rmse_to_LOO_surrogate_mean":
                    "H_max_surrogate_LOO_rmse"
            }
        )
    )

    cal = bc.merge(
        max_loo,
        on=["regime_plot", "block"],
        how="left",
        validate="one_to_one",
    )

    if cal["H_max_surrogate_LOO_rmse"].isna().any():
        raise ValueError("Failed to match some block-level surrogate LOO maxima.")

    all_original_exceed = bool(
        np.all(
            cal["H_rmse_original_to_surrogate_mean"].to_numpy(float)
            >
            cal["H_max_surrogate_LOO_rmse"].to_numpy(float)
        )
    )

    # Order calibration rows exactly as panel (b).
    cal_rows = []

    for regime in ORDER:
        for block in sorted(
            cal.loc[cal["regime_plot"] == regime, "block"].unique()
        ):
            row = cal[
                (cal["regime_plot"] == regime)
                & (cal["block"] == block)
            ].iloc[0]

            cal_rows.append(row)

    cal_plot = pd.DataFrame(cal_rows).reset_index(drop=True)
    cal_plot["row"] = np.arange(len(cal_plot))


    # ============================================================
    # Figure
    # ============================================================

    fig = plt.figure(figsize=(13.2, 8.2))

    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 1.18],
        width_ratios=[1.13, 1.0],
        hspace=0.31,
        wspace=0.36,
    )

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])


    # ------------------------------------------------------------
    # (a) Scale-resolved regime means
    # ------------------------------------------------------------

    for regime in ORDER:
        d = regime_curves[regime]

        ax_a.plot(
            d["L"],
            d["mean"],
            linewidth=1.6,
            label=DISPLAY[regime],
        )

        ax_a.fill_between(
            d["L"],
            d["min"],
            d["max"],
            alpha=0.12,
        )

    ax_a.axhline(
        0.0,
        linewidth=0.9,
        linestyle="--",
        alpha=0.65,
    )

    ax_a.set_xlim(
        float(L_grid.min()),
        float(L_grid.max()),
    )

    ax_a.set_xlabel(r"Window length, $L$")
    ax_a.set_ylabel(
        r"$H_{\mathrm{original}}(L)-"
        r"\overline{H}_{\mathrm{shuffle}}(L)$"
    )

    ax_a.set_title(
        "(a) Scale-resolved entropy displacement",
        loc="left",
    )

    ax_a.legend(
        frameon=False,
        ncol=5,
        fontsize=8.2,
        loc="lower left",
    )


    # ------------------------------------------------------------
    # (b) Complete displacement map
    # ------------------------------------------------------------

    im = ax_b.imshow(
        disp,
        aspect="auto",
        origin="upper",
        extent=[
            float(L_grid.min()),
            float(L_grid.max()),
            len(row_labels) - 0.5,
            -0.5,
        ],
    )

    ax_b.set_xlabel(r"Window length, $L$")
    ax_b.set_ylabel("Riemann-zero configuration")

    ax_b.set_title(
        "(b) Configuration–scale displacement",
        loc="left",
    )

    ax_b.set_yticks(group_centres)
    ax_b.set_yticklabels([SHORT[r] for r in ORDER])

    for y in group_boundaries:
        ax_b.axhline(
            y,
            linewidth=0.8,
            alpha=0.55,
        )

    cb = fig.colorbar(
        im,
        ax=ax_b,
        fraction=0.046,
        pad=0.04,
    )

    cb.set_label(r"$H_{\mathrm{original}}-\overline{H}_{\mathrm{shuffle}}$")


    # ------------------------------------------------------------
    # (c) Paired leave-one-out calibration
    # ------------------------------------------------------------

    y = cal_plot["row"].to_numpy()

    x_loo = cal_plot[
        "H_max_surrogate_LOO_rmse"
    ].to_numpy(float)

    x_orig = cal_plot[
        "H_rmse_original_to_surrogate_mean"
    ].to_numpy(float)

    # Connector for each configuration.
    ax_c.hlines(
        y,
        x_loo,
        x_orig,
        linewidth=1.0,
        alpha=0.45,
    )

    # Two endpoint sets. Their vertical coordinates are identical.
    ax_c.scatter(
        x_loo,
        y,
        s=38,
        marker="o",
        label="Maximum within-surrogate discrepancy",
        zorder=3,
    )

    ax_c.scatter(
        x_orig,
        y,
        s=42,
        marker="D",
        label="Original vs surrogate mean",
        zorder=4,
    )

    ax_c.set_ylim(
        len(cal_plot) - 0.5,
        -0.5,
    )

    ax_c.set_xlabel("Entropy RMS discrepancy")

    ax_c.set_title(
        "(c) Within-surrogate calibration",
        loc="left",
    )

    ax_c.set_yticks(group_centres)
    ax_c.set_yticklabels([SHORT[r] for r in ORDER])

    for yline in group_boundaries:
        ax_c.axhline(
            yline,
            linewidth=0.8,
            alpha=0.40,
        )

    ax_c.legend(
        frameon=False,
        fontsize=8.2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )


    # ============================================================
    # Layout and export
    # ============================================================

    fig.subplots_adjust(
        left=0.085,
        right=0.965,
        top=0.965,
        bottom=0.12,
    )

    stem = OUT / "Figure4_gap_ordering"

    fig.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


    # ============================================================
    # Numerical audit files
    # ============================================================

    audit_rows = []

    for regime in ORDER:
        sb = bc[bc["regime_plot"] == regime]

        audit_rows.append(
            {
                "regime": regime,
                "mean_H_RMSE":
                    sb["H_rmse_original_to_surrogate_mean"].mean(),
                "min_H_RMSE":
                    sb["H_rmse_original_to_surrogate_mean"].min(),
                "max_H_RMSE":
                    sb["H_rmse_original_to_surrogate_mean"].max(),
                "mean_signed_H_difference":
                    sb["H_mean_signed_difference"].mean(),
            }
        )

    pd.DataFrame(audit_rows).to_csv(
        OUT / "Figure4_regime_audit.csv",
        index=False,
    )

    cal_plot[
        [
            "regime_plot",
            "block",
            "H_max_surrogate_LOO_rmse",
            "H_rmse_original_to_surrogate_mean",
            "H_surrogate_LOO_rmse_percentile",
            "H_empirical_upper_tail",
        ]
    ].to_csv(
        OUT / "Figure4_block_calibration.csv",
        index=False,
    )

    print("SUCCESS: final Figure 4 generated.")
    print(
        "Fraction of configuration-scale entropy differences "
        f"< -1e-12: {negative_fraction:.4%}"
    )
    print(
        "All 20 original entropy RMSE values exceed the "
        "maximum within-block surrogate LOO RMSE: "
        f"{all_original_exceed}"
    )
    print(f"Outputs: {OUT.resolve()}")


# ============================================================
# FIGURE 5
# ============================================================
def generate_figure5():

    from pathlib import Path
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


    # ============================================================
    # Paths
    # ============================================================

    BASE = Path(".")

    POINTS_FILE = BASE / "entropy_variance_points.csv"
    PAIRS_FILE = BASE / "near_equal_variance_pairs.csv"
    PMF_EXAMPLES_FILE = BASE / "near_equal_variance_pmf_examples.csv"
    PMF_MASSES_FILE = BASE / "selected_pmf_masses.csv"

    OUT = BASE / "figure5_final_output"
    OUT.mkdir(parents=True, exist_ok=True)


    # ============================================================
    # Load data
    # ============================================================

    pts = pd.read_csv(POINTS_FILE)
    pairs = pd.read_csv(PAIRS_FILE)
    examples = pd.read_csv(PMF_EXAMPLES_FILE)
    masses = pd.read_csv(PMF_MASSES_FILE)

    required_pts = {
        "role",
        "regime",
        "block",
        "L",
        "H",
        "number_variance",
        "massey_bound",
        "massey_slack",
        "observation_id",
    }

    required_examples = {
        "example_rank",
        "selection_tolerance",
        "L",
        "obs1",
        "obs2",
        "H1",
        "H2",
        "variance1",
        "variance2",
        "abs_entropy_difference",
        "abs_variance_difference",
        "pmf_total_variation_distance",
    }

    required_masses = {
        "example_rank",
        "which_observation",
        "observation_id",
        "L",
        "k",
        "p",
    }

    for name, frame, cols in [
        ("entropy_variance_points", pts, required_pts),
        ("near_equal_variance_pmf_examples", examples, required_examples),
        ("selected_pmf_masses", masses, required_masses),
    ]:
        missing = cols - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")


    # ============================================================
    # Regime labels
    # ============================================================

    ORDER = ["low", "midlow", "1e12", "1e21", "1e22"]

    DISPLAY = {
        "low": r"First $10^4$ zeros",
        "midlow": r"$45{,}001$–$55{,}000$",
        "1e12": r"Near $10^{12}$",
        "1e21": r"Near $10^{21}$",
        "1e22": r"Near $10^{22}$",
    }


    def canon(v):
        s = str(v).strip().lower()
        c = re.sub(r"[\s_\-–—]+", "", s)

        if "1e22" in c or "10^22" in c:
            return "1e22"
        if "1e21" in c or "10^21" in c:
            return "1e21"
        if "1e12" in c or "10^12" in c:
            return "1e12"
        if "midlow" in c or "45001" in c or "robust" in c:
            return "midlow"
        if "low" in c or c in {"first", "first10000", "first10k"}:
            return "low"
        return s


    pts = pts.copy()
    pts["regime_plot"] = pts["regime"].map(canon)

    unknown = set(pts["regime_plot"]) - set(ORDER)
    if unknown:
        raise ValueError(f"Unrecognized regime labels: {sorted(unknown)}")


    # ============================================================
    # Audit key analysis quantities
    # ============================================================

    n_obs = len(pts)
    min_slack = float(pts["massey_slack"].min())
    violations = int((pts["massey_slack"] < -1e-10).sum())

    # Strictest near-equal-variance example
    example = examples.sort_values(
        ["selection_tolerance", "example_rank"]
    ).iloc[0]

    example_rank = int(example["example_rank"])
    example_L = float(example["L"])
    H1 = float(example["H1"])
    H2 = float(example["H2"])
    V1 = float(example["variance1"])
    V2 = float(example["variance2"])
    dH = float(example["abs_entropy_difference"])
    dV = float(example["abs_variance_difference"])
    tv = float(example["pmf_total_variation_distance"])

    relative_variance_difference = dV / ((V1 + V2) / 2.0)


    # ============================================================
    # PMF preparation
    # ============================================================

    m1 = masses[
        (masses["example_rank"] == example_rank)
        & (masses["which_observation"] == 1)
    ].sort_values("k")

    m2 = masses[
        (masses["example_rank"] == example_rank)
        & (masses["which_observation"] == 2)
    ].sort_values("k")

    ks = np.array(sorted(set(m1["k"]) | set(m2["k"])), dtype=int)

    p1_map = dict(zip(m1["k"].astype(int), m1["p"].astype(float)))
    p2_map = dict(zip(m2["k"].astype(int), m2["p"].astype(float)))

    p1 = np.array([p1_map.get(int(k), 0.0) for k in ks], dtype=float)
    p2 = np.array([p2_map.get(int(k), 0.0) for k in ks], dtype=float)

    dp = p1 - p2


    # ============================================================
    # Figure layout
    # ============================================================

    fig = plt.figure(figsize=(13.4, 8.3))

    outer = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 0.96],
        width_ratios=[1.0, 1.0],
        hspace=0.31,
        wspace=0.28,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])

    lower = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer[1, :],
        height_ratios=[2.8, 0.9],
        hspace=0.06,
    )

    ax_c = fig.add_subplot(lower[0])
    ax_d = fig.add_subplot(lower[1], sharex=ax_c)


    # ============================================================
    # Panel (a): Entropy–variance phase portrait
    # ============================================================

    for regime in ORDER:
        sub = pts[pts["regime_plot"] == regime]

        ax_a.scatter(
            sub["number_variance"],
            sub["H"],
            s=10,
            alpha=0.22,
            label=DISPLAY[regime],
            rasterized=True,
        )

    vmax = float(pts["number_variance"].max()) * 1.03
    vgrid = np.linspace(0.0, vmax, 600)

    bound = 0.5 * np.log(
        2.0 * np.pi * np.e * (vgrid + 1.0 / 12.0)
    )

    ax_a.plot(
        vgrid,
        bound,
        linewidth=2.0,
        label="Entropy–variance upper bound",
        zorder=5,
    )

    ax_a.set_xlabel(r"Number variance, $\Sigma^2$")
    ax_a.set_ylabel(r"Count entropy, $H$")
    ax_a.set_title(
        "(a) Empirical entropy–variance relation",
        loc="left",
    )

    pearson_r = float(pts["H"].corr(pts["number_variance"], method="pearson"))
    spearman_rho = float(pts["H"].corr(pts["number_variance"], method="spearman"))

    ax_a.text(
        0.035, 0.95,
        rf"$r={pearson_r:.4f}$" + "\n" + rf"$\rho={spearman_rho:.6f}$",
        transform=ax_a.transAxes, ha="left", va="top", fontsize=9.0,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                  edgecolor="0.75", alpha=0.92),
    )

    ax_a.legend(
        frameon=False,
        fontsize=8.0,
        ncol=2,
        loc="lower right",
        columnspacing=1.0,
        handletextpad=0.5,
    )


    # ============================================================
    # Panel (b): Entropy separation under variance matching
    # ============================================================

    tol = (
        pairs[pairs["rank_within_tolerance"] == 1]
        .sort_values("relative_variance_tolerance")
        .copy()
    )
    tol_x = 100.0 * tol["relative_variance_tolerance"].to_numpy(float)
    tol_y = tol["abs_entropy_difference"].to_numpy(float)

    ax_b.plot(tol_x, tol_y, marker="o", linewidth=1.8, markersize=5.5)

    ax_b.set_xlabel(r"Relative variance tolerance, $\varepsilon$ (%)")
    ax_b.set_ylabel(r"Maximum observed entropy separation, $\max|\Delta H|$")
    ax_b.set_title(
        "(b) Maximum entropy separation under variance matching",
        loc="left",
    )
    ax_b.set_xticks(tol_x)
    ax_b.set_ylim(bottom=0)


    # ============================================================
    # Panel (c): Near-equal-variance PMFs
    # ============================================================

    offset = 0.055

    ax_c.vlines(
        ks - offset,
        0,
        p1,
        linewidth=1.5,
    )

    ax_c.scatter(
        ks - offset,
        p1,
        s=38,
        marker="o",
        label=r"Near $10^{12}$, block 4",
        zorder=4,
    )

    ax_c.vlines(
        ks + offset,
        0,
        p2,
        linewidth=1.5,
    )

    ax_c.scatter(
        ks + offset,
        p2,
        s=38,
        marker="D",
        label=r"Near $10^{22}$, block 2",
        zorder=4,
    )

    ax_c.set_ylabel(r"$p_L(k)$")

    ax_c.set_title(
        rf"(c) Count laws under near-equal variance ($L={example_L:g}$)",
        loc="left",
    )

    ax_c.legend(
        frameon=False,
        fontsize=8.0,
        ncol=2,
        loc="upper left",
        columnspacing=1.0,
        handletextpad=0.5,
    )

    # Compact numerical annotation
    annotation = (
        rf"$\Sigma_1^2={V1:.6f},\quad \Sigma_2^2={V2:.6f}$"
        "\n"
        rf"relative difference $={100*relative_variance_difference:.3f}\%$"
        "\n"
        rf"$H_1={H1:.6f},\quad H_2={H2:.6f}$"
        "\n"
        rf"$|\Delta H|={dH:.6f},\quad d_{{\rm TV}}={tv:.5f}$"
    )

    ax_c.text(
        0.985,
        0.94,
        annotation,
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=8.6,
    )

    ax_c.tick_params(
        axis="x",
        labelbottom=False,
    )


    # ============================================================
    # Residual strip: p1(k) - p2(k)
    # ============================================================

    ax_d.axhline(
        0.0,
        linestyle="--",
        linewidth=0.8,
        alpha=0.60,
    )

    ax_d.vlines(
        ks,
        0,
        dp,
        linewidth=1.4,
    )

    ax_d.scatter(
        ks,
        dp,
        s=28,
    )

    ax_d.set_xlabel(r"Window count, $k$")
    ax_d.set_ylabel(r"$p_1(k)-p_2(k)$")
    ax_d.set_xticks(ks)


    # ============================================================
    # Layout / export
    # ============================================================

    fig.subplots_adjust(
        left=0.08,
        right=0.975,
        top=0.965,
        bottom=0.085,
    )

    stem = OUT / "Figure5_entropy_variance_FINAL_TRUE"

    fig.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


    # ============================================================
    # Audit CSV
    # ============================================================

    audit = pd.DataFrame(
        {
            "quantity": [
                "num_original_observations",
                "pearson_r",
                "spearman_rho",
                "minimum_massey_slack",
                "massey_violations_below_minus_1e-10",
                "example_rank",
                "example_L",
                "variance1",
                "variance2",
                "relative_variance_difference",
                "H1",
                "H2",
                "absolute_entropy_difference",
                "pmf_total_variation_distance",
            ],
            "value": [
                n_obs,
                pearson_r,
                spearman_rho,
                min_slack,
                violations,
                example_rank,
                example_L,
                V1,
                V2,
                relative_variance_difference,
                H1,
                H2,
                dH,
                tv,
            ],
        }
    )

    audit.to_csv(
        OUT / "Figure5_audit_FINAL_TRUE.csv",
        index=False,
    )

    print("SUCCESS: publication Figure 5 generated.")
    print(f"Original observations: {n_obs}")
    print(f"Minimum entropy-bound slack: {min_slack:.12f}")
    print(f"Bound violations below -1e-10: {violations}")
    print(
        "Near-equal-variance example: "
        f"L={example_L}, "
        f"relative variance difference={100*relative_variance_difference:.5f}%, "
        f"|ΔH|={dH:.6f}, "
        f"dTV={tv:.6f}"
    )
    print(f"Outputs: {OUT.resolve()}")


def parse_master_args():
    parser = argparse.ArgumentParser(
        description="Generate publication Figures 1–5 from the combined master script."
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5],
        help="Figure numbers to generate (default: all figures).",
    )
    return parser.parse_args()


def main():
    args = parse_master_args()
    selected = args.figures

    generators = {
        1: generate_figure1,
        2: generate_figure2,
        3: generate_figure3,
        4: generate_figure4,
        5: generate_figure5,
    }

    print("\n" + "=" * 68)
    print("MASTER MANUSCRIPT FIGURE GENERATION")
    print(f"Selected figures: {selected}")
    print("=" * 68)

    failures = []
    for figure_number in selected:
        print(f"\n>>> STARTING FIGURE {figure_number} <<<")
        try:
            generators[figure_number]()
            print(f"<<< FIGURE {figure_number} COMPLETED >>>")
        except Exception as exc:
            failures.append((figure_number, exc))
            print(f"<<< FIGURE {figure_number} FAILED: {exc} >>>", file=sys.stderr)

    print("\n" + "=" * 68)
    if failures:
        print("GENERATION FINISHED WITH ERRORS")
        for figure_number, exc in failures:
            print(f"  Figure {figure_number}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
    else:
        print("ALL SELECTED FIGURES GENERATED SUCCESSFULLY")
    print("=" * 68)


if __name__ == "__main__":
    main()
