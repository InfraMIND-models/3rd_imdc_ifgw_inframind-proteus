"""LSTM sequence-to-scalar season-grain regressor (BaseModel).

A **hybrid** model: an LSTM encodes the leakage-safe pre-season *weekly incidence sequence*
(all weeks with epiweek <= t0 = EW25 of the season's start year, i.e. the run-up to the
season, which has not yet started at the issue point), and the recurrent embedding is
concatenated with a dense head over the **tabular season matrix** (the same leakage-safe
season features the other tabular models use). One model per target; predictions are returned
on the natural target scale (any log1p train transform inverted inside the model).

Rationale for the hybrid: a pure sequence model on incidence alone mostly relearns the
seasonal autocorrelation that climatology/SARIMAX already capture. Bolting the LSTM embedding
onto the tabular features tests the sharper question — does the *raw temporal shape* of the
run-up add anything over the engineered tabular summaries? Uncertainty via shadow-model
split-conformal intervals (MC-dropout fallback); a coarse driver story via block-permutation
importance (sequence vs each tabular block).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .base_model import BaseModel


# ---------------------------------------------------------------- torch network
def _build_net(n_seq_feat: int, n_tab: int):
    import torch.nn as nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            h = config.LSTM_HIDDEN
            self.lstm = nn.LSTM(n_seq_feat, h, num_layers=config.LSTM_LAYERS,
                                batch_first=True,
                                dropout=config.LSTM_DROPOUT if config.LSTM_LAYERS > 1 else 0.0)
            self.tab = nn.Sequential(nn.Linear(n_tab, h), nn.ReLU(),
                                     nn.Dropout(config.LSTM_DROPOUT)) if n_tab else None
            head_in = h + (h if n_tab else 0)
            self.head = nn.Sequential(nn.Linear(head_in, h), nn.ReLU(),
                                      nn.Dropout(config.LSTM_DROPOUT), nn.Linear(h, 1))

        def forward(self, seq, tab):
            out, _ = self.lstm(seq)
            z = out[:, -1, :]                         # last timestep == t0 (sequences are right-aligned)
            if self.tab is not None:
                z = __import__("torch").cat([z, self.tab(tab)], dim=1)
            return self.head(z).squeeze(-1)

    return _Net()


class LSTMModel(BaseModel):
    supports_uncertainty = True       # split-conformal intervals, MC-dropout fallback

    def __init__(self, target: str, cat_features: list[str] | None = None,
                 quantiles=config.LSTM_QUANTILES, repo=None, **_):
        self.name = "lstm"
        self.target = target
        self.repo = repo
        self.log_target = target in config.LOG1P_TARGETS
        self.round_target = target == "peak_timing_week"
        self._declared_cats = list(cat_features or [])
        self.quantiles = tuple(quantiles)
        self._series_ = None

    # -------------------------------------------------- leakage-safe weekly sequences
    def _unit_series(self) -> dict:
        """unit -> (epiweek[], log1p incidence[]) sorted chronologically (built once)."""
        if self._series_ is not None:
            return self._series_
        if self.repo is None:
            raise ValueError("LSTMModel needs a DataRepository (repo=) to build sequences")
        uc = self.repo.unit_col
        p = self.repo.panel()[[uc, "epiweek", "incidence"]].dropna(subset=["epiweek"])
        p = p.sort_values([uc, "epiweek"])
        series = {}
        for u, g in p.groupby(uc, sort=False):
            inc = np.log1p(np.clip(g["incidence"].to_numpy(float), 0.0, None))
            series[u] = (g["epiweek"].to_numpy(np.int64), inc.astype(np.float32))
        self._series_ = series
        return series

    def _sequences(self, index) -> tuple[np.ndarray, np.ndarray]:
        """(N, L, 3) tensor [log-incidence, sin(week), cos(week)] + (N, L) valid mask.

        Right-aligned: the final timestep is t0; short histories are front zero-padded.
        Strict leakage cut: only weeks with epiweek <= season*100 + SEASON_ISSUE_WEEK.
        """
        series = self._unit_series()
        L = config.LSTM_LOOKBACK
        seq = np.zeros((len(index), L, 3), dtype=np.float32)
        valid = np.zeros((len(index), L), dtype=bool)
        for i, (u, s) in enumerate(index):
            ew, inc = series.get(u, (np.empty(0, np.int64), np.empty(0, np.float32)))
            if ew.size == 0:
                continue
            t0 = int(s) * 100 + config.SEASON_ISSUE_WEEK
            m = ew <= t0
            e, v = ew[m][-L:], inc[m][-L:]
            n = len(v)
            if n == 0:
                continue
            wk = (e % 100).astype(np.float32)
            seq[i, L - n:, 0] = v
            seq[i, L - n:, 1] = np.sin(2 * np.pi * wk / 52.0)
            seq[i, L - n:, 2] = np.cos(2 * np.pi * wk / 52.0)
            valid[i, L - n:] = True
        return seq, valid

    def _scale_seq(self, seq: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Standardize the log-incidence channel (train stats); re-zero padded positions."""
        out = seq.copy()
        out[..., 0] = (out[..., 0] - self.seq_mean_) / self.seq_std_
        out[~valid] = 0.0                              # keep padding inert after scaling
        return out

    # -------------------------------------------------- tabular design (numeric + one-hot)
    def _encode_fit(self, X: pd.DataFrame) -> None:
        self.cat_features_ = [c for c in self._declared_cats if c in X.columns]
        self.num_features_ = [c for c in X.columns if c not in self.cat_features_]
        med = X[self.num_features_].median()
        self.medians_ = {c: (0.0 if pd.isna(med.get(c)) else float(med[c])) for c in self.num_features_}
        self.cat_levels_ = {c: sorted(map(str, X[c].dropna().unique())) for c in self.cat_features_}
        self.col_source_ = list(self.num_features_)
        for c in self.cat_features_:
            self.col_source_ += [c] * len(self.cat_levels_[c])

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        cols = [X[c].fillna(self.medians_[c]).to_numpy(float) for c in self.num_features_]
        for c in self.cat_features_:
            s = X[c].astype("object").where(X[c].notna(), None).map(
                lambda v: str(v) if v is not None else None)
            for lv in self.cat_levels_[c]:
                cols.append((s == lv).to_numpy(float))
        return np.column_stack(cols) if cols else np.zeros((len(X), 0))

    # -------------------------------------------------- target scaling / inversion
    def _fit_y(self, yv: np.ndarray) -> np.ndarray:
        yt = np.log1p(yv) if self.log_target else yv.astype(float)
        self.y_mean_, self.y_std_ = float(np.mean(yt)), float(np.std(yt) or 1.0)
        return (yt - self.y_mean_) / self.y_std_

    def _inv_y(self, z: np.ndarray) -> np.ndarray:
        p = z * self.y_std_ + self.y_mean_
        if self.log_target:
            p = np.expm1(p)
        p = np.clip(p, 0.0, None)
        if self.round_target:
            p = np.round(p)
        if self.target == "peak_timing_week":
            p = np.clip(p, 1.0, None)
        return p

    # -------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y: pd.Series, *,
            cat_features: list[str] | None = None) -> "LSTMModel":
        import torch
        if cat_features is not None:
            self._declared_cats = list(cat_features)
        torch.manual_seed(config.LSTM_SEED)
        np.random.seed(config.LSTM_SEED)

        yv = pd.Series(np.asarray(y, float), index=X.index)
        keep = yv.notna().to_numpy()
        Xk = X.loc[keep]
        self._encode_fit(Xk)
        self.feature_names_ = list(X.columns)

        seq, valid = self._sequences(Xk.index)
        v0 = seq[..., 0][valid]
        self.seq_mean_ = float(v0.mean()) if v0.size else 0.0
        self.seq_std_ = float(v0.std() or 1.0)
        S = self._scale_seq(seq, valid)

        from sklearn.preprocessing import StandardScaler
        M = self._design(Xk)
        self.x_scaler_ = StandardScaler().fit(M) if M.shape[1] else None
        T = self.x_scaler_.transform(M) if self.x_scaler_ is not None else M

        z = self._fit_y(yv.loc[keep].to_numpy(float))
        truth = yv.loc[keep].to_numpy(float)
        seasons = Xk.index.get_level_values("season").to_numpy()
        self.nat_cap_ = 3.0 * float(np.nanmax(truth))         # tame extrapolation

        # leakage-safe early stopping: hold out the latest train season
        tr, va = self._es_split(seasons)
        self.net_, best_val = self._train_net(S, T, z, tr, va)
        self.diagnostics_ = {"n_train": int(tr.sum()), "n_val": int(va.sum()),
                             "val_mse": round(best_val, 4), "n_features_tab": int(T.shape[1])}
        # cache a held-out slice for permutation importance (cheap, leakage-safe)
        self._imp_cache_ = (S[va], T[va], truth[va])

        # split-conformal residual quantiles (natural scale). Leaves the point model untouched
        # -> only the intervals change vs MC-dropout.
        self.resid_q_ = None
        if config.LSTM_CONFORMAL:
            self._fit_conformal(S, T, z, truth, seasons, va)
        return self

    # -------------------------------------------------- training helpers
    @staticmethod
    def _es_split(seasons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Boolean (train, val) masks: hold out the latest season for early stopping."""
        uniq = np.unique(seasons)
        if len(uniq) >= 2:
            va = seasons == uniq.max()
        else:
            va = np.zeros(len(seasons), bool); va[:max(1, len(seasons) // 5)] = True
        return ~va, va

    def _train_net(self, S: np.ndarray, T: np.ndarray, z: np.ndarray,
                   tr: np.ndarray, va: np.ndarray):
        """Fit one network with leakage-safe early stopping; return (net, best_val_mse)."""
        import torch
        net = _build_net(S.shape[2], T.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=config.LSTM_LR,
                               weight_decay=config.LSTM_WEIGHT_DECAY)
        lossf = torch.nn.MSELoss()
        tS, tT, tz = (torch.tensor(S[tr]), torch.tensor(T[tr], dtype=torch.float32),
                      torch.tensor(z[tr], dtype=torch.float32))
        vS, vT, vz = (torch.tensor(S[va]), torch.tensor(T[va], dtype=torch.float32),
                      torch.tensor(z[va], dtype=torch.float32))
        n = len(tz); bs = config.LSTM_BATCH
        best_val, best_state, since, loss = np.inf, None, 0, None
        for _ in range(config.LSTM_MAX_EPOCHS):
            net.train()
            for b in torch.randperm(n).split(bs):
                opt.zero_grad()
                loss = lossf(net(tS[b], tT[b]), tz[b])
                loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vloss = float(lossf(net(vS, vT), vz)) if len(vz) else float(loss)
            if vloss < best_val - 1e-4:
                best_val, best_state, since = vloss, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                since += 1
                if since >= config.LSTM_PATIENCE:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best_val

    def _fit_conformal(self, S: np.ndarray, T: np.ndarray, z: np.ndarray,
                       truth: np.ndarray, seasons: np.ndarray, va: np.ndarray) -> None:
        """Split-conformal residual quantiles.

        Preferred: a *shadow* net trained on the earlier seasons gives honest out-of-sample
        residuals on the last K held-out seasons (the main net trains on all data and is left
        untouched -> only the band is calibrated). Falls back to single-season residuals from
        the main net's early-stop hold-out (optimistic), then to MC-dropout if even that is tiny.
        """
        lo, hi = self.quantiles[0], self.quantiles[-1]
        uniq = np.sort(np.unique(seasons))
        mode, resid = None, None
        # shadow-model multi-season calibration: need >=K cal seasons + >=2 to train+early-stop the shadow
        if len(uniq) >= config.LSTM_CONFORMAL_SEASONS + 2:
            K = config.LSTM_CONFORMAL_SEASONS
            cal_seasons = uniq[-K:]
            cal = np.isin(seasons, cal_seasons)
            rest_idx = np.flatnonzero(~cal)                   # earlier seasons -> shadow train + early-stop
            rtr, rva = self._es_split(seasons[rest_idx])
            tr_mask = np.zeros(len(seasons), bool); tr_mask[rest_idx[rtr]] = True
            va_mask = np.zeros(len(seasons), bool); va_mask[rest_idx[rva]] = True
            cal_net, _ = self._train_net(S, T, z, tr_mask, va_mask)
            pred = self._forward_arrays(S[cal], T[cal], train_mode=False, net=cal_net)
            resid, mode = truth[cal] - pred, f"shadow_{K}season"
            self.diagnostics_["conformal_cal_seasons"] = [int(s) for s in cal_seasons]
        elif va.sum() >= 2:                                   # single held-out season (optimistic)
            pred = self._forward_arrays(S[va], T[va], train_mode=False)
            resid, mode = truth[va] - pred, "single_season"
        if resid is None or len(resid) < 2:                   # too small -> MC-dropout fallback
            return
        self.resid_q_ = {lo: float(np.quantile(resid, lo)), 0.5: float(np.median(resid)),
                         hi: float(np.quantile(resid, hi))}
        self.diagnostics_["conformal_mode"] = mode
        self.diagnostics_["conformal_n"] = int(len(resid))
        self.diagnostics_["conformal_width90"] = round(self.resid_q_[hi] - self.resid_q_[lo], 3)

    # -------------------------------------------------- predict
    def _forward_arrays(self, S: np.ndarray, T: np.ndarray, train_mode: bool = False,
                        net=None) -> np.ndarray:
        """Natural-scale point prediction from already-scaled sequence/tabular arrays.

        `net` defaults to the fitted main net; pass a shadow net for conformal calibration.
        """
        import torch
        net = net if net is not None else self.net_
        net.train(train_mode)                          # train_mode=True -> dropout on (MC)
        with torch.no_grad():
            z = net(torch.tensor(S), torch.tensor(T, dtype=torch.float32)).cpu().numpy()
        return np.minimum(self._inv_y(z), self.nat_cap_)

    def _forward(self, X: pd.DataFrame, train_mode: bool = False) -> np.ndarray:
        seq, valid = self._sequences(X.index)
        S = self._scale_seq(seq, valid)
        M = self._design(X)
        T = self.x_scaler_.transform(M) if self.x_scaler_ is not None else M
        return self._forward_arrays(S, T, train_mode)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._forward(X, train_mode=False)

    def _mc_quantiles(self, X: pd.DataFrame, quantiles) -> np.ndarray:
        draws = np.stack([self._forward(X, train_mode=True)
                          for _ in range(config.LSTM_MC_SAMPLES)], axis=0)
        return np.quantile(draws, quantiles, axis=0).T          # (N, n_quantiles)

    def predict_quantiles(self, X: pd.DataFrame, quantiles=None) -> pd.DataFrame:
        quantiles = tuple(quantiles or self.quantiles)
        if getattr(self, "resid_q_", None) is not None:         # split-conformal: point + residual band
            point = self.predict(X)
            band = {q: point + self.resid_q_[q] for q in self.resid_q_}
            arr = np.column_stack([band[q] for q in quantiles])
        else:                                                   # fallback: MC-dropout
            arr = self._mc_quantiles(X, quantiles)
        arr = np.clip(arr, 0.0, None)
        if self.target == "peak_timing_week":
            arr = np.clip(arr, 1.0, None)
        arr = np.sort(arr, axis=1)
        return pd.DataFrame(arr, index=X.index, columns=[float(q) for q in quantiles])

    # -------------------------------------------------- block-permutation importance
    def feature_importance(self, X: pd.DataFrame | None = None,
                           y: pd.Series | None = None) -> pd.DataFrame:
        import torch
        S, T, truth = self._imp_cache_
        if len(truth) < 3:
            return pd.DataFrame(columns=["feature", "importance"])
        if len(truth) > config.LSTM_IMPORTANCE_MAXROWS:
            idx = np.random.default_rng(0).choice(len(truth), config.LSTM_IMPORTANCE_MAXROWS, False)
            S, T, truth = S[idx], T[idx], truth[idx]
        rng = np.random.default_rng(0)
        self.net_.eval()

        def mae(Sm, Tm):
            with torch.no_grad():
                z = self.net_(torch.tensor(Sm), torch.tensor(Tm, dtype=torch.float32)).cpu().numpy()
            return float(np.mean(np.abs(np.minimum(self._inv_y(z), self.nat_cap_) - truth)))

        base = mae(S, T)
        # group tabular columns by source feature/block; the sequence is its own block
        blocks: dict[str, list[int]] = {}
        for j, src in enumerate(self.col_source_):
            blocks.setdefault(src, []).append(j)
        rows = []
        # sequence block: shuffle whole sequences across samples
        Sp = S[rng.permutation(len(S))]
        rows.append(("weekly_sequence", mae(Sp, T) - base))
        for src, cols in blocks.items():
            Tp = T.copy()
            Tp[:, cols] = Tp[rng.permutation(len(Tp))][:, cols]
            rows.append((src, mae(S, Tp) - base))
        out = pd.DataFrame(rows, columns=["feature", "importance"])
        return out.sort_values("importance", ascending=False, ignore_index=True)
