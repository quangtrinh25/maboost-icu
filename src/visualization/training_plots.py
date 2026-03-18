"""
src/visualization/training_plots.py
=====================================
Evaluation plots for MaBoost — mortality + LOS.

Mortality: roc_curve, pr_curve, confusion_matrix, calibration, threshold_sweep
LOS:       scatter, residuals, error_by_quartile
Benchmark: bar chart comparing all models
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                              confusion_matrix, f1_score, mean_absolute_error,
                              mean_squared_error, precision_recall_curve,
                              roc_auc_score, roc_curve)

C = {"blue":"#2563EB","red":"#DC2626","green":"#16A34A","gray":"#6B7280","amber":"#D97706"}

plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,
                     "axes.grid":True,"grid.alpha":0.25,"font.size":10})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {'/'.join(path.parts[-2:])}")


# ===========================================================================
# Mortality
# ===========================================================================

def plot_roc(y_true, y_prob, out_dir, n_boot=500):
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    rng  = np.random.default_rng(42); tprs=[]; base=np.linspace(0,1,101)
    for _ in range(n_boot):
        idx=rng.integers(0,len(y_true),len(y_true))
        if y_true[idx].sum()==0: continue
        f,t,_=roc_curve(y_true[idx],y_prob[idx])
        tprs.append(np.interp(base,f,t))
    lo,hi=np.percentile(tprs,2.5,0),np.percentile(tprs,97.5,0)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot(fpr,tpr,C["blue"],lw=2,label=f"MaBoost  AUC={auc:.4f}")
    ax.fill_between(base,lo,hi,alpha=0.15,color=C["blue"],label="95% CI")
    ax.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.4)
    ax.set(xlabel="FPR",ylabel="TPR",title="ROC — in-hospital mortality",xlim=(0,1),ylim=(0,1)); ax.legend()
    _save(fig, Path(out_dir)/"mortality"/"roc_curve.png")


def plot_pr(y_true, y_prob, out_dir):
    ap=average_precision_score(y_true,y_prob)
    prec,rec,_=precision_recall_curve(y_true,y_prob)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot(rec,prec,C["blue"],lw=2,label=f"AP={ap:.4f}")
    ax.axhline(y_true.mean(),color=C["gray"],ls="--",lw=0.8,label=f"No-skill ({y_true.mean():.3f})")
    ax.set(xlabel="Recall",ylabel="Precision",title="PR curve — mortality",xlim=(0,1),ylim=(0,1)); ax.legend()
    _save(fig, Path(out_dir)/"mortality"/"pr_curve.png")


def plot_confusion(y_true, y_prob, out_dir):
    thrs=np.linspace(0.01,0.99,200)
    f1s=[f1_score(y_true,(y_prob>=t).astype(int),zero_division=0) for t in thrs]
    t=float(thrs[np.argmax(f1s)])
    yp=(y_prob>=t).astype(int); cm=confusion_matrix(y_true,yp)
    fig,ax=plt.subplots(figsize=(4.5,4))
    im=ax.imshow(cm,cmap="Blues"); plt.colorbar(im,ax=ax,fraction=0.046)
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Survived","Died"],
           yticklabels=["Survived","Died"],xlabel="Predicted",ylabel="Actual",
           title=f"Confusion matrix  (thr={t:.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f"{cm[i,j]:,}\n({100*cm[i,j]/cm.sum():.1f}%)",ha="center",va="center",
                    color="white" if cm[i,j]>cm.max()/2 else "black",fontsize=9)
    _save(fig, Path(out_dir)/"mortality"/"confusion_matrix.png")


def plot_calibration(y_true, y_prob, out_dir):
    frac,mp=calibration_curve(y_true,y_prob,n_bins=10,strategy="uniform")
    fig,(a1,a2)=plt.subplots(2,1,figsize=(6,7),gridspec_kw={"height_ratios":[3,1]})
    a1.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.5); a1.plot(mp,frac,"o-",color=C["blue"],ms=4,lw=1.8)
    a1.set(xlabel="Mean predicted prob",ylabel="Fraction positives",title="Calibration — mortality",xlim=(0,1),ylim=(0,1))
    a2.hist(y_prob[y_true==0],bins=40,alpha=0.6,color=C["blue"],label="Survived")
    a2.hist(y_prob[y_true==1],bins=40,alpha=0.6,color=C["red"],label="Died")
    a2.set(xlabel="Predicted probability",ylabel="Count"); a2.legend()
    plt.tight_layout(); _save(fig, Path(out_dir)/"mortality"/"calibration.png")


def plot_threshold_sweep(y_true, y_prob, out_dir):
    thrs=np.linspace(0.01,0.99,200); sens,spec,f1v,ppv=[],[],[],[]
    for t in thrs:
        yp=(y_prob>=t).astype(int)
        tp=((yp==1)&(y_true==1)).sum(); tn=((yp==0)&(y_true==0)).sum()
        fp=((yp==1)&(y_true==0)).sum(); fn=((yp==0)&(y_true==1)).sum()
        sens.append(tp/max(tp+fn,1)); spec.append(tn/max(tn+fp,1))
        f1v.append(f1_score(y_true,yp,zero_division=0)); ppv.append(tp/max(tp+fp,1))
    bt=float(thrs[np.argmax(f1v)])
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(thrs,sens,C["blue"],lw=1.8,label="Sensitivity")
    ax.plot(thrs,spec,C["green"],lw=1.8,label="Specificity")
    ax.plot(thrs,f1v,C["red"],lw=1.8,label="F1")
    ax.plot(thrs,ppv,C["gray"],lw=1.5,ls="--",label="PPV")
    ax.axvline(bt,color="k",ls=":",lw=1,label=f"Best F1 ({bt:.2f})")
    ax.set(xlabel="Threshold",ylabel="Metric",title="Threshold analysis",xlim=(0,1),ylim=(0,1)); ax.legend(ncol=2)
    _save(fig, Path(out_dir)/"mortality"/"threshold_sweep.png")


# ===========================================================================
# LOS
# ===========================================================================

def plot_los_scatter(y_true, y_pred, out_dir):
    fig,ax=plt.subplots(figsize=(6,5))
    ax.scatter(np.log1p(y_true),np.log1p(y_pred),alpha=0.15,s=6,color=C["blue"])
    lim=np.log1p(max(y_true.max(),y_pred.max()))+0.2
    ax.plot([0,lim],[0,lim],"k--",lw=0.8)
    mae=mean_absolute_error(y_true,y_pred)
    ax.text(0.05,0.92,f"MAE={mae:.2f}d",transform=ax.transAxes,fontsize=9)
    ax.set(xlabel="Actual LOS log(1+d)",ylabel="Predicted LOS log(1+d)",title="LOS prediction")
    _save(fig, Path(out_dir)/"los"/"scatter.png")


def plot_los_residuals(y_true, y_pred, out_dir):
    err=y_pred-y_true
    fig,ax=plt.subplots(figsize=(7,4))
    ax.hist(np.clip(err,-15,15),bins=80,color=C["blue"],alpha=0.8,edgecolor="none")
    ax.axvline(0,color="k",lw=1); ax.axvline(err.mean(),color=C["red"],lw=1.5,ls="--",label=f"bias={err.mean():.2f}d")
    ax.set(xlabel="Error (pred − actual) days",title="LOS residuals"); ax.legend()
    _save(fig, Path(out_dir)/"los"/"residuals.png")


def plot_los_by_quartile(y_true, y_pred, out_dir):
    q=np.percentile(y_true,[0,25,50,75,100]); lbs,maes=[],[]
    for i in range(4):
        m=(y_true>=q[i])&(y_true<q[i+1])
        if m.sum()==0: continue
        lbs.append(f"Q{i+1}\n({q[i]:.1f}–{q[i+1]:.1f}d)"); maes.append(mean_absolute_error(y_true[m],y_pred[m]))
    fig,ax=plt.subplots(figsize=(6,4))
    ax.bar(lbs,maes,color=C["blue"],alpha=0.82)
    for x,v in enumerate(maes): ax.text(x,v+0.05,f"{v:.2f}",ha="center",fontsize=9)
    ax.set(ylabel="MAE (days)",title="LOS error by quartile")
    _save(fig, Path(out_dir)/"los"/"error_by_quartile.png")


# ===========================================================================
# Benchmark bar chart
# ===========================================================================

def plot_benchmark(results, out_dir):
    rs=sorted(results,key=lambda r:r.auroc)
    names=[r.name for r in rs]; aucs=[r.auroc for r in rs]
    colors=[C["blue"] if "MaBoost" in n else C["gray"] for n in names]
    fig,ax=plt.subplots(figsize=(9,max(5,len(names)*0.55)))
    bars=ax.barh(names,aucs,color=colors,alpha=0.85)
    for bar,v in zip(bars,aucs):
        ax.text(v+0.003,bar.get_y()+bar.get_height()/2,f"{v:.4f}",va="center",fontsize=8.5)
    ax.set(xlabel="AUROC",title="Mortality prediction — model comparison",xlim=(0.4,1.02))
    _save(fig, Path(out_dir)/"benchmark_comparison.png")


# ===========================================================================
# Training history
# ===========================================================================

def plot_training_history(losses: list, aucs: list, out_dir: str):
    ep=list(range(1,len(losses)+1))
    fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4))
    a1.plot(ep,losses,C["blue"],lw=1.8); a1.set(xlabel="Epoch",ylabel="Loss",title="Training loss")
    a2.plot(ep,aucs,C["green"],lw=1.8); a2.axhline(max(aucs),color=C["gray"],ls=":",lw=0.8,
            label=f"Best={max(aucs):.4f}"); a2.set(xlabel="Epoch",ylabel="AUROC",title="Val AUROC"); a2.legend()
    plt.tight_layout(); _save(fig, Path(out_dir)/"training_history.png")


# ===========================================================================
# Convenience: save everything
# ===========================================================================

def save_all(y_mort_true, y_mort_prob, y_los_true, y_los_pred,
             benchmark_results=None, losses=None, aucs=None, out_dir="results"):
    print("\n[Plots] Saving evaluation figures …")
    plot_roc(y_mort_true, y_mort_prob, out_dir)
    plot_pr(y_mort_true, y_mort_prob, out_dir)
    plot_confusion(y_mort_true, y_mort_prob, out_dir)
    plot_calibration(y_mort_true, y_mort_prob, out_dir)
    plot_threshold_sweep(y_mort_true, y_mort_prob, out_dir)
    plot_los_scatter(y_los_true, y_los_pred, out_dir)
    plot_los_residuals(y_los_true, y_los_pred, out_dir)
    plot_los_by_quartile(y_los_true, y_los_pred, out_dir)
    if benchmark_results:
        plot_benchmark(benchmark_results, out_dir)
    if losses and aucs:
        plot_training_history(losses, aucs, out_dir)
    print(f"[Plots] Done → {out_dir}/")