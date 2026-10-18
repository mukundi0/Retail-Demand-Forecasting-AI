"""Reproducible chronological experiments for 28-day M5 demand forecasting."""
from __future__ import annotations
import argparse, hashlib, json, platform
from datetime import datetime, timezone
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

SEED, HORIZON, FOLDS, TRAIN_DAYS = 42, 28, 3, 365
STATIC = ["store_id", "cat_id", "dept_id", "state_id"]
CODES = ["store_code", "category_code", "department_code", "state_code"]
DYNAMIC = ["lag_1","lag_7","lag_14","lag_28","mean_7","mean_28","std_28","zero_share_28",
           "weekday","month","weekend","event_day","snap_CA","snap_TX","snap_WI"]
FEATURES = DYNAMIC + CODES
ERROR_COLUMNS = {
    "seasonal_naive_7d": "absolute_error_seasonal_naive_7d",
    "four_week_weekday_mean": "absolute_error_four_week_weekday_mean",
    "lightgbm_recursive": "absolute_error_lightgbm_recursive",
}

def load_data(data_dir, max_series):
    sales_file, calendar_file = data_dir/"sales_train_validation.csv", data_dir/"calendar.csv"
    if not sales_file.exists() or not calendar_file.exists():
        raise FileNotFoundError(f"Expected {sales_file} and {calendar_file}; see README for data setup.")
    sales = pd.read_csv(sales_file)
    days = sorted([c for c in sales.columns if c.startswith("d_") and c[2:].isdigit()], key=lambda x:int(x[2:]))
    if len(days) < 364 + HORIZON*FOLDS:
        raise ValueError("Need at least 52 complete weeks plus three 28-day validation folds.")
    if max_series and max_series > 0 and len(sales) > max_series:
        sales = sales.sample(n=max_series, random_state=SEED).sort_index().reset_index(drop=True)
    meta = sales[["id"]+STATIC].copy()
    codes = pd.DataFrame({c:pd.Categorical(meta[s]).codes.astype(np.int32) for c,s in zip(CODES,STATIC)})
    y = sales[days].to_numpy(dtype=np.float32)
    cal = pd.read_csv(calendar_file).set_index("d").reindex(days).reset_index()
    if cal["date"].isna().any(): raise ValueError("Calendar does not cover every sales day.")
    date = pd.to_datetime(cal["date"])
    calfeat = pd.DataFrame({
      "weekday":date.dt.dayofweek.astype(np.int8), "month":date.dt.month.astype(np.int8),
      "weekend":(date.dt.dayofweek>=5).astype(np.int8),
      "event_day":(cal.get("event_name_1",pd.Series(index=cal.index,dtype=object)).notna() |
                   cal.get("event_name_2",pd.Series(index=cal.index,dtype=object)).notna()).astype(np.int8),
      **{c:cal.get(c,pd.Series(0,index=cal.index)).fillna(0).astype(np.int8) for c in ["snap_CA","snap_TX","snap_WI"]}
    })
    return meta,y,codes,days,calfeat,sales_file

def volatility(history):
    weeks = history[:,-364:].reshape(len(history),52,7).sum(axis=2)
    pos = weeks > 0
    count = pos.sum(axis=1)
    mean = np.divide((weeks*pos).sum(axis=1),count,out=np.zeros(len(history)),where=count>0)
    var = np.divide(np.where(pos,(weeks-mean[:,None])**2,0).sum(axis=1),count,
                    out=np.zeros(len(history)),where=count>0)
    cv = np.divide(np.sqrt(var),mean,out=np.zeros(len(history)),where=mean>0)
    rank = pd.Series(cv).rank(method="average",pct=True).to_numpy()
    labels = np.where(rank<=1/3,"low",np.where(rank<=2/3,"moderate","high"))
    return labels,(weeks==0).mean(axis=1)

def make_features(history, target_days, cal, codes):
    n = history.shape[0]
    result = {}
    for lag in (1,7,14,28): result[f"lag_{lag}"] = history[:,target_days-lag].T.reshape(-1)
    prev = history[:,target_days[:,None]-np.arange(1,29)]
    result["mean_7"] = prev[:,:,:7].mean(axis=2).T.reshape(-1)
    result["mean_28"] = prev.mean(axis=2).T.reshape(-1)
    result["std_28"] = prev.std(axis=2).T.reshape(-1)
    result["zero_share_28"] = (prev==0).mean(axis=2).T.reshape(-1)
    for c in ["weekday","month","weekend","event_day","snap_CA","snap_TX","snap_WI"]:
        result[c] = np.repeat(cal.iloc[target_days][c].to_numpy(),n)
    for c in CODES: result[c] = np.tile(codes[c].to_numpy(),len(target_days))
    return pd.DataFrame(result,columns=FEATURES)

def fit(history, cutoff, cal, codes):
    first = max(28,cutoff-TRAIN_DAYS+1)
    valid_start = max(first+1,cutoff-27)
    tr_days = np.arange(first,valid_start,dtype=np.int32)
    va_days = np.arange(valid_start,cutoff+1,dtype=np.int32)
    xtr = make_features(history,tr_days,cal,codes)
    ytr = history[:,tr_days].T.reshape(-1)
    train = lgb.Dataset(xtr,label=ytr,categorical_feature=CODES,free_raw_data=True)
    xva = make_features(history,va_days,cal,codes)
    yva = history[:,va_days].T.reshape(-1)
    valid = lgb.Dataset(xva,label=yva,categorical_feature=CODES,reference=train,free_raw_data=True)
    return lgb.train({"objective":"regression_l1","metric":"mae","learning_rate":.05,
      "num_leaves":31,"min_data_in_leaf":100,"feature_fraction":.9,"verbosity":-1,
      "seed":SEED,"feature_fraction_seed":SEED,"bagging_seed":SEED,"data_random_seed":SEED,
      "num_threads":-1},train,num_boost_round=300,valid_sets=[valid],
      callbacks=[lgb.early_stopping(30,verbose=False)])

def recursive(model, history, cutoff, cal, codes):
    work = history[:,:cutoff+1].copy()
    forecasts=[]
    for day in range(cutoff+1,cutoff+HORIZON+1):
        x=make_features(work,np.array([work.shape[1]-1]),cal.iloc[:work.shape[1]],codes)
        for c in ["weekday","month","weekend","event_day","snap_CA","snap_TX","snap_WI"]:
            x[c]=cal.iloc[day][c]
        pred=np.maximum(model.predict(x,num_threads=-1),0).astype(np.float32)
        forecasts.append(pred); work=np.column_stack([work,pred])
    return np.stack(forecasts,axis=1)

def bootstrap_comparisons(series_metrics, replicates):
    rng = np.random.default_rng(SEED)
    rows = []
    for group in ("all", "low", "moderate", "high"):
        data = series_metrics if group == "all" else series_metrics[series_metrics["volatility_group"] == group]
        columns = ["actual_demand_sum", "forecast_observations", *ERROR_COLUMNS.values()]
        by_series = data.groupby("series_id", as_index=False)[columns].sum()
        if by_series.empty:
            continue
        denominator = by_series["actual_demand_sum"].to_numpy(dtype=np.float64)
        model_errors = {name:by_series[column].to_numpy(dtype=np.float64)
                        for name,column in ERROR_COLUMNS.items()}
        model_wape = model_errors["lightgbm_recursive"].sum()/denominator.sum()
        sampled = rng.integers(0, len(by_series), size=(replicates, len(by_series)))
        sample_denominator = denominator[sampled].sum(axis=1)
        valid = sample_denominator > 0
        for baseline in ("seasonal_naive_7d", "four_week_weekday_mean"):
            baseline_wape = model_errors[baseline].sum()/denominator.sum()
            improvement = (
                model_errors[baseline][sampled[valid]].sum(axis=1)
                - model_errors["lightgbm_recursive"][sampled[valid]].sum(axis=1)
            ) / sample_denominator[valid]
            rows.append({
                "volatility_group":group,
                "baseline_model":baseline,
                "series":int(len(by_series)),
                "bootstrap_replicates":int(len(improvement)),
                "pooled_lightgbm_WAPE":float(model_wape),
                "pooled_baseline_WAPE":float(baseline_wape),
                "WAPE_improvement":float(baseline_wape-model_wape),
                "improvement_95CI_low":float(np.quantile(improvement,0.025)),
                "improvement_95CI_high":float(np.quantile(improvement,0.975)),
                "relative_improvement_pct":float((baseline_wape-model_wape)/baseline_wape*100) if baseline_wape else float("nan"),
                "bootstrap_probability_improvement_positive":float((improvement>0).mean()),
            })
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=Path("data/raw"))
    p.add_argument("--output-dir",type=Path,default=Path("artifacts"))
    p.add_argument("--max-series",type=int,default=1000,help="Seeded sample size; set 0 for all series.")
    p.add_argument("--bootstrap-replicates",type=int,default=1000)
    args=p.parse_args()
    if args.max_series < 0 or args.bootstrap_replicates < 1:
        p.error("--max-series must be zero or positive and --bootstrap-replicates must be positive.")
    meta,y,codes,days,cal,sales_file=load_data(args.data_dir,args.max_series)
    starts=[y.shape[1]-HORIZON*FOLDS+i*HORIZON for i in range(FOLDS)]
    rows=[]; per_series=[]
    for fold,start in enumerate(starts,1):
        cutoff=start-1; actual=y[:,start:start+HORIZON]; hist=y[:,:cutoff+1]
        labels,zero_share=volatility(hist)
        naive=np.tile(hist[:,-7:],(1,4))
        weekday=np.tile(hist[:,-28:].reshape(len(hist),4,7).mean(axis=1),(1,4))
        model=fit(hist,cutoff,cal,codes)
        preds={"seasonal_naive_7d":naive,"four_week_weekday_mean":weekday,
               "lightgbm_recursive":recursive(model,hist,cutoff,cal,codes)}
        for name,pred in preds.items():
            per_series.append(pd.DataFrame({
                "fold":fold,"series_id":meta["id"].to_numpy(),"volatility_group":labels,
                "actual_demand_sum":actual.sum(axis=1),"forecast_observations":actual.shape[1],
                "model":name,"absolute_error_sum":np.abs(actual-pred).sum(axis=1),
            }))
            for group in ("all","low","moderate","high"):
                mask=np.ones(len(labels),dtype=bool) if group=="all" else labels==group
                error=np.abs(actual[mask]-pred[mask])
                error_sum=float(error.sum())
                demand_sum=float(actual[mask].sum())
                wape=error_sum/demand_sum if demand_sum else float("nan")
                mae=float(error.mean())
                rows.append({"fold":fold,"train_through":days[cutoff],"validation_start":days[start],
                  "validation_end":days[start+HORIZON-1],"model":name,"volatility_group":group,
                  "series":int(mask.sum()),"WAPE":wape,"MAE":mae,
                  "absolute_error_sum":error_sum,"actual_demand_sum":demand_sum,
                  "forecast_observations":int(error.size),
                  "median_zero_week_share":float(np.median(zero_share[mask]))})
        print(f"Completed fold {fold}/{FOLDS}: {days[start]} to {days[start+HORIZON-1]}")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fold_metrics=pd.DataFrame(rows)
    (args.output_dir/"metrics.csv").write_text(fold_metrics.to_csv(index=False),encoding="utf-8")
    pooled=(fold_metrics.groupby(["model","volatility_group"],as_index=False)
      .agg(absolute_error_sum=("absolute_error_sum","sum"),
           actual_demand_sum=("actual_demand_sum","sum"),
           forecast_observations=("forecast_observations","sum"),
           folds=("fold","nunique"),
           mean_fold_WAPE=("WAPE","mean"),mean_fold_MAE=("MAE","mean")))
    pooled["pooled_WAPE"]=np.divide(pooled["absolute_error_sum"],pooled["actual_demand_sum"],
      out=np.full(len(pooled),np.nan),where=pooled["actual_demand_sum"].to_numpy()>0)
    pooled["pooled_MAE"]=pooled["absolute_error_sum"]/pooled["forecast_observations"]
    (args.output_dir/"pooled_metrics.csv").write_text(pooled.to_csv(index=False),encoding="utf-8")
    per_series_long=pd.concat(per_series,ignore_index=True)
    per_series_wide=(per_series_long.pivot(
      index=["fold","series_id","volatility_group","actual_demand_sum","forecast_observations"],
      columns="model",values="absolute_error_sum").reset_index())
    per_series_wide.columns.name=None
    per_series_wide=per_series_wide.rename(columns=ERROR_COLUMNS)
    (args.output_dir/"series_metrics.csv").write_text(per_series_wide.to_csv(index=False),encoding="utf-8")
    bootstrap=bootstrap_comparisons(per_series_wide,args.bootstrap_replicates)
    (args.output_dir/"bootstrap_comparison.csv").write_text(bootstrap.to_csv(index=False),encoding="utf-8")
    manifest={"research_topic":"Evaluating the Effectiveness of AI-Based Demand Forecasting for Retail Businesses Under Volatile Demand Conditions",
      "seed":SEED,"horizon_days":HORIZON,"folds":FOLDS,"training_window_days":TRAIN_DAYS,
      "max_series":args.max_series,"series_evaluated":len(meta),
      "bootstrap_replicates":args.bootstrap_replicates,
      "sales_file_sha256":hashlib.sha256(sales_file.read_bytes()).hexdigest(),
      "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
      "lightgbm":lgb.__version__,"run_timestamp_utc":datetime.now(timezone.utc).isoformat(),
      "volatility_definition":"terciles of CV among positive-demand weekly totals over the preceding 52 complete weeks; zero-week share reported separately"}
    (args.output_dir/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Saved fold, pooled, per-series, and bootstrap metrics to {args.output_dir}")

if __name__=="__main__": main()

