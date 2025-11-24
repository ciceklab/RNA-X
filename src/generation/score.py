from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import math
import re
import numpy as np


BASES = set("AUGC")
RC_MAP = str.maketrans("AUGCaugc", "UACGuacg")

def reverse_complement(seq: str) -> str:
    return seq.translate(RC_MAP)[::-1]

def is_valid_rna(seq: str) -> bool:
    return all(c in BASES for c in seq)

def softplus(x: float) -> float:
    return math.log1p(math.exp(x))

def clip01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def entropy01(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    H = -(p*math.log(p) + (1-p)*math.log(1-p))
    return H / math.log(2)

def kmer_entropy(seq: str, k: int = 3) -> float:
    if len(seq) < k:
        return 0.0
    from collections import Counter
    counts = Counter(seq[i:i+k] for i in range(len(seq)-k+1))
    total = sum(counts.values())
    H = 0.0
    for c in counts.values():
        p = c/total
        H -= p*math.log2(p + 1e-12)
    maxH = math.log2(min(4**k, len(seq)-k+1))
    return 0.0 if maxH <= 0 else H/maxH

def longest_homopolymer_run(seq: str) -> int:
    best = 0; cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            cur += 1
        else:
            best = max(best, cur); cur = 1
    return max(best, cur)

def gc_fraction(seq: str) -> float:
    return sum(1 for c in seq if c in "GC")/max(1,len(seq))


import RNA as _RNA

def fold_mfe_dotbracket(seq: str) -> Tuple[str, float, str]:
    fc = _RNA.fold_compound(seq)
    db, mfe = fc.mfe()
    return db, float(mfe), "vienna"

def parse_dot_bracket(db: str) -> Dict[str, Any]:
    n=len(db)
    stack=[]; pairs={}
    for i,c in enumerate(db):
        if c=='(':
            stack.append(i)
        elif c==')' and stack:
            j=stack.pop()
            pairs[j]=i
    pair_set=set((i,j) if i<j else (j,i) for i,j in pairs.items())
    # stems
    used=set(); stems=[]
    for i,j in sorted(pair_set):
        if (i,j) in used: continue
        stem=[(i,j)]; used.add((i,j))
        ii, jj = i, j
        while (ii+1,jj-1) in pair_set and (ii+1,jj-1) not in used:
            ii+=1; jj-=1; stem.append((ii,jj)); used.add((ii,jj))
        stems.append(stem)
    # hairpins: simple hairpins where innermost pair encloses only dots
    hairpins=[]
    for stem in stems:
        ii, jj = stem[-1]
        loop = list(range(ii+1, jj))
        if loop and all(db[k]=='.' for k in loop):
            hairpins.append(loop)
    # unpaired runs
    runs=[]; s=None
    for i,c in enumerate(db):
        if c=='.':
            if s is None: s=i
        else:
            if s is not None:
                runs.append((s,i-1)); s=None
    if s is not None: runs.append((s,len(db)-1))
    return {"pairs": pair_set, "stems": stems, "hairpins": hairpins, "runs": runs}

def loop_exposure_from_db(db: str) -> float:
    parsed = parse_dot_bracket(db)
    if parsed["hairpins"]:
        loop = max(parsed["hairpins"], key=len)
        exp = np.mean([1.0 for _ in loop]) 
        return float(exp)  # == 1.0
    if parsed["runs"]:
        run = max(parsed["runs"], key=lambda x: x[1]-x[0])
        length = run[1]-run[0]+1
        return float(length/max(1,len(db)))
    return float(db.count('.')/max(1,len(db)))

def compactness_from_db(db: str) -> Dict[str, float]:
    P = parse_dot_bracket(db)
    stems = P["stems"]; runs = P["runs"]
    stem_count = len(stems)
    mean_stem_len = float(np.mean([len(s) for s in stems])) if stems else 0.0
    longest_ss = max((e-s+1 for s,e in runs), default=0)
    total_unpaired = sum(e-s+1 for s,e in runs)
    # Map to [0,1] with simple rules
    score = 0.0
    if stem_count >= 1: score += 0.4
    if mean_stem_len >= 2: score += 0.3
    if longest_ss >= 10: score -= 0.3
    score = clip01(score+0.3)  # small base
    return {
        "COMP": score,
        "stem_count": stem_count,
        "mean_stem_len": round(mean_stem_len,2),
        "longest_ss_run": longest_ss,
        "total_unpaired": total_unpaired
    }

def low_complexity_penalty(seq: str) -> Tuple[float, int, float]:
    ent = kmer_entropy(seq, k=3)
    homo = longest_homopolymer_run(seq)
    pen = 0.0
    if homo >= 7:
        pen = 0.15
    else:
        pen = 0.15*(1.0 - ent)
    return float(pen), homo, float(ent)

def max_contig_rc_match(seq: str) -> int:
    rc = reverse_complement(seq)
    best = 0
    for i in range(len(seq)):
        for j in range(len(rc)):
            r=0
            while i+r<len(seq) and j+r<len(rc) and seq[i+r]==rc[j+r]:
                r+=1
            if r>best: best=r
    return best

def self_assoc_penalty(seq: str) -> Tuple[float, int]:
    m = max_contig_rc_match(seq)
    if m >= 14: pen = 0.2
    elif m >= 10: pen = 0.1 + 0.02*(m-10)  # 0.1..0.18
    else: pen = 0.0
    return float(min(pen,0.2)), m

G4_REGEX = re.compile(r"(G{3,}\w{1,7}){3}G{3,}", re.IGNORECASE)
def g4_flag(seq: str) -> bool:
    return bool(G4_REGEX.search(seq))

def immuno_penalty(seq: str, db: Optional[str]) -> Tuple[float, float, int, float]:
    gu = sum(1 for i in range(len(seq)-1) if seq[i:i+2] in ("GU","UG"))/max(1,(len(seq)-1))
    u_run = longest_homopolymer_run(''.join([c if c=='U' else 'X' for c in seq]))
    long_ds_frac = 0.0
    if db:
        P=parse_dot_bracket(db)
        long_stems=[s for s in P["stems"] if len(s)>=10]
        long_ds_frac = sum(len(s) for s in long_stems)/max(1,len(seq))
    pen = min(1.0, 0.6*gu + 0.2*max(0,(u_run-5)/10.0) + 0.2*min(1.0,long_ds_frac))
    return 0.1*pen, gu, u_run, long_ds_frac  # scale to ≤0.1

HHR_CORE = re.compile(r"(CUGA|GUGA)", re.IGNORECASE)
HDV_CORE = re.compile(r"CUGGG", re.IGNORECASE)
TWISTER_CORE = re.compile(r"(AGUA|GAAA|UCGA)", re.IGNORECASE)  
HAIRPIN_CORE = re.compile(r"(GUC|AUGU|UGUA)", re.IGNORECASE)   

def ribozyme_screen(seq: str, db: str) -> Tuple[bool, float, Dict[str, Any]]:
    hits = {
        "hammerhead": len(HHR_CORE.findall(seq)),
        "hdv": len(HDV_CORE.findall(seq)),
        "twister": len(TWISTER_CORE.findall(seq)),
        "hairpin": len(HAIRPIN_CORE.findall(seq))
    }
    strong_family = None
    for fam, n in hits.items():
        if n >= 2: strong_family = fam; break
    P=parse_dot_bracket(db)
    tri_stem = (len(P["stems"])>=3)
    has_pseudoknot_proxy = (len(P["stems"])>=2)
    strong = (strong_family is not None)
    veto=False; pen=0.0
    if strong:
        if strong_family=="hammerhead" and tri_stem:
            veto=True
        elif strong_family=="hdv" and has_pseudoknot_proxy:
            veto=True
        elif strong_family in ("twister","hairpin") and len(P["stems"])>=2:
            veto=True
        else:
            pen=0.3
    else:
        weak = any(n==1 for n in hits.values())
        if weak: pen=0.15
    return veto, float(pen), {"hits":hits, "tri_stem":tri_stem}

KOZAK_STRONG = lambda ctx: (ctx[-3] in "AG") and (ctx[+4]=='G')
SD_REGEX = re.compile(r"AGGAGG|AGGAG|GGAGG", re.IGNORECASE)
PAS_REGEX = re.compile(r"AAUAAA|AUUAAA", re.IGNORECASE)

def find_orfs(seq: str, start_pos: int) -> int:
    stops = {"UAA","UAG","UGA"}
    codons = 0
    for i in range(start_pos, len(seq)-2, 3):
        codons += 1
        if seq[i:i+3] in stops:
            return codons
    return 0  # no stop found

def translation_penalties(seq: str, cellular_profile: bool=True) -> Tuple[bool, float, Dict[str, Any]]:
    veto=False; pen=0.0
    tags=[]
    for i in range(len(seq)-2):
        codon = seq[i:i+3]
        if codon=="AUG":
            left = seq[i-3:i] if i-3>=0 else "N"*(3-i)+seq[:i]
            right = seq[i+3:i+7] if i+7<=len(seq) else seq[i+3:]+ "N"*(i+7-len(seq))
            ctx = left + "AUG" + right
            strong = ( (ctx[2] in "AG") and (ctx[7]=='G') )
            orflen = find_orfs(seq, i)
            if strong and orflen >= 20 and cellular_profile:
                veto=True; tags.append(f"kozak_strong_orf={orflen}")
            elif orflen >= 20 and cellular_profile:
                pen += 0.3; tags.append(f"kozak_orf={orflen}")
            elif orflen >= 10 and cellular_profile:
                pen += 0.15; tags.append(f"kozak_orf={orflen}")
    for m in SD_REGEX.finditer(seq):
        sd_end = m.end()
        for i in range(sd_end+4, min(sd_end+10, len(seq)-2)):
            if seq[i:i+3]=="AUG":
                orflen = find_orfs(seq, i)
                if orflen >= 20 and cellular_profile:
                    veto=True; tags.append(f"sd_orf={orflen}")
                elif orflen >= 10 and cellular_profile:
                    pen += 0.15; tags.append(f"sd_orf={orflen}")
    for i in range(len(seq)-2):
        if seq[i:i+3]=="AUG":
            start=max(0,i-50); end=min(len(seq), i+50)
            window = seq[start:end]
            if re.search(r"[CU]{8,}", window):
                pen += 0.05; tags.append("ppt_near_aug")
    if PAS_REGEX.search(seq) and cellular_profile:
        pen += 0.05; tags.append("PAS")
    if re.search(r"A{10,}", seq) and cellular_profile:
        pen += 0.05; tags.append("polyA10")
    return veto, float(min(pen,0.3)), {"tags": tags}

def normalized_mfe_score(seq: str, mfe: float, baseline_stats: Optional[Dict[int, Tuple[float,float]]] = None,
                         z_band: Tuple[float,float]=(-1.5, -0.2)) -> Tuple[float, Dict[str, Any]]:
    L = len(seq)
    if baseline_stats and L in baseline_stats and baseline_stats[L][1] > 1e-6:
        mu, sd = baseline_stats[L]
        z = (mfe - mu)/sd
        lo, hi = z_band
        if lo <= z <= hi:
            stab=1.0
        elif z > hi:
            stab = math.exp(-1.0*(z - hi))
            stab = math.exp(-1.0*(lo - z))
        return clip01(stab), {"norm":"z", "z":round(z,3), "mu":round(mu,3), "sd":round(sd,3)}
    per_nt = mfe / max(1,L)
    lo, hi = -0.40, -0.20
    if lo <= per_nt <= hi:
        stab=1.0
    elif per_nt > hi:
        stab = math.exp(-5.0*(per_nt - hi))  # too weak (less negative)
    else:
        stab = math.exp(-3.0*(lo - per_nt))  # too rigid (more negative)
    return clip01(stab), {"norm":"per_nt", "per_nt":round(per_nt,3)}

@dataclass
class ScoreConfig:
    gc_low: float = 0.35
    gc_high: float = 0.65
    wS: float = 0.4  # stability weight
    wH: float = 0.4  # shape weight
    cellular_profile: bool = True  # enable translation/immuno for in vivo/cellular
    baseline_mfe_stats: Optional[Dict[int, Tuple[float,float]]] = None  # optional length->(mu,sd)

class Scorer:
    def __init__(self, cfg: Optional[ScoreConfig]=None):
        self.cfg = cfg or ScoreConfig()

    def score_sequence(self, prediction_model, protein_seq: str, rna_seq: str) -> Dict[str, Any]:
        # Validate
        if not is_valid_rna(rna_seq[4:]):
            return {"final": 0.0, "veto": True, "veto_reason": "invalid_alphabet", "id": None}


        import torch
        logits = prediction_model([protein_seq], [rna_seq])
        # print("logits:", logits)
        probs  = torch.sigmoid(logits)
        H = entropy01(probs)
        CONF = 1.0 - H
        BIND_star = probs * (0.7 + 0.3*CONF)

        rna_seq = rna_seq[4:]

        db, mfe, engine = fold_mfe_dotbracket(rna_seq)
        STAB, stab_meta = normalized_mfe_score(rna_seq, mfe, baseline_stats=self.cfg.baseline_mfe_stats)

        LOOP = loop_exposure_from_db(db)
        LOOP_map = 0.0 if LOOP<=0.2 else (1.0 if LOOP>=0.6 else (LOOP-0.2)/(0.6-0.2))
        comp_obj = compactness_from_db(db)
        COMP = comp_obj["COMP"]

        GC = gc_fraction(rna_seq)
        GC_BIN = 1.0 if (self.cfg.gc_low <= GC <= self.cfg.gc_high) else 0.0

        SHAPE = (0.5*LOOP_map + 0.3*COMP + 0.2*GC_BIN)

        P_LC, homo_run, k3_ent = low_complexity_penalty(rna_seq)
        P_SELF, rc_match = self_assoc_penalty(rna_seq)
        g4 = g4_flag(rna_seq)
        P_G4 = 0.1 if (g4 and self.cfg.cellular_profile) else 0.0
        P_IMM, gu_density, u_run, long_ds = immuno_penalty(rna_seq, db if self.cfg.cellular_profile else None)
        if not self.cfg.cellular_profile:
            P_IMM = 0.0
        RIBO_veto, P_RIBO, ribo_meta = ribozyme_screen(rna_seq, db)
        TX_veto, P_TX, tx_meta = translation_penalties(rna_seq, cellular_profile=self.cfg.cellular_profile)
        veto_reason=None
        if RIBO_veto:
            veto_reason="ribozyme_topology"
        if TX_veto and veto_reason is None:
            veto_reason="translation_strong_start"
        if homo_run >= 9 and veto_reason is None:
            veto_reason="homopolymer>=9"
        if rc_match >= 14 and veto_reason is None:
            veto_reason="self_dimer_extreme"

        if veto_reason is not None:
            return {
                "final": 0.0,
                "veto": True,
                "veto_reason": veto_reason,
                "p_bind": float(probs),
                "engine": engine,
                "mfe": float(mfe),
                "db": db,
                "rna_seq": rna_seq
            }

        P_TOTAL = min(0.60, P_SELF + P_LC + P_G4 + P_IMM + P_RIBO + P_TX)
        SPEC = 1.0 - P_TOTAL

        wS = self.cfg.wS; wH = self.cfg.wH
        FINAL = BIND_star * (wS*STAB + (1-wS)) * (wH*SHAPE + (1-wH)) * SPEC
        FINAL = clip01(FINAL)

        return {
            "final": float(FINAL),
            "veto": False,
            "p_bind": float(probs),
            "bind_conf": float(CONF),
            "BIND_star": float(BIND_star),
            "engine": engine,
            "mfe": float(mfe),
            "stab": float(STAB),
            "stab_meta": stab_meta,
            "db": db,
            "loop": float(LOOP_map),
            "comp": float(COMP),
            "gc_frac": float(GC),
            "gc_bin": float(GC_BIN),
            "shape": float(SHAPE),
            "penalties": {
                "P_SELF": float(P_SELF),
                "rc_match": int(rc_match),
                "P_LC": float(P_LC),
                "homo_run": int(homo_run),
                "k3_entropy": float(k3_ent),
                "P_G4": float(P_G4),
                "g4_flag": bool(g4),
                "P_IMM": float(P_IMM),
                "gu_density": float(gu_density),
                "u_run": int(u_run),
                "long_ds_frac": float(long_ds),
                "P_RIBO": float(P_RIBO),
                "ribo_meta": ribo_meta,
                "P_TX": float(P_TX),
                "tx_meta": tx_meta,
                "P_TOTAL": float(P_TOTAL)
            }
        }

def score_with_details(prediction_model, protein_seq: str, rna_seq: str,
                       cfg: Optional[ScoreConfig]=None) -> Dict[str, Any]:
    return Scorer(cfg).score_sequence(prediction_model, protein_seq, rna_seq)


socrecfg = ScoreConfig()
def value_of_seq(prediction_model, protein_seq: str, rna_seq: str, use_scoring:bool=True, cfg: Optional[ScoreConfig]=socrecfg) -> float:
    if use_scoring:
        scorer = Scorer(cfg)
        out = scorer.score_sequence(prediction_model, protein_seq, rna_seq)
        return float(out["final"]), out
    else:
        return 1, None
