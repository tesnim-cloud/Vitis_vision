# -*- coding: utf-8 -*-
# Les importe
import os
os.environ.setdefault("QT_QPA_PLATFORMTHEME", "none")       
import sys, json, math, time, csv, warnings
import numpy as np
import cv2

# ---------------------------------------------------------------------------------------------------------
# Constantes de DÉFINITION (identiques pour toutes les branches — ce ne sont pas des réglages)
# ---------------------------------------------------------------------------------------------------------
BORD_NOEUD_PCT = 30.0      # limites du nœud : base + 30 % × (pic − base)
COUPE_APP_PCT = 50.0       # appendices plus fins que 50 % du diamètre (vrille, pétiole) ignorés pour trouver la tige
MARGE_BOUTS = 0.8          # bouts de la branche ignorés : 0,8 × diamètre à chaque bout (coupes)
N_MOYENNE = 15             # images moyennées avec la touche « a »

# ---------------------------------------------------------------------------------------------------------
# Réglages de SEGMENTATION : ceux de segmentation_branche.py qui donnent un bon résultat sur la vraie branche
# ---------------------------------------------------------------------------------------------------------
EXPO_US = 7000.0           # exposition (µs) — modifiable avec + / -
FLOU = 10                  # flou 2×10+1 = 21×21 (pour le MASQUE seulement)
MORPHO = 3                 # curseur « Morpho » = 3 → noyau 2×3+1 = 7×7
AIRE_MIN = 5000            # curseur « Aire min x100 » = 50 → 5000 px²
SEUIL_BAS_MIN = 12         # écart de couleur minimal avec le fond pour qu'un pixel CLAIR (bois pâle, bourgeon) compte
CONTRASTE_MIN = 15         # bord accepté si fond et branche diffèrent d'au moins 15 (1 = laisse passer des bords faux : bourgeon, vrille)


# =========================================================================================================
# 1. CALIBRATION (lecture seule : calibration.json écrit par calibration_mm.py)
# =========================================================================================================
class Calibration:
    """D_mm = a·D_px + b pour les diamètres ; L_mm = a·L_px pour les longueurs. Sans fichier : pixels."""
    def __init__(self, chemin):
        self.ok, self.a, self.b, self.K, self.dist, self._maps = False, 1.0, 0.0, None, None, None
        if os.path.exists(chemin):
            d = json.load(open(chemin, encoding="utf-8"))
            if "mm_par_px" in d:
                self.ok, self.a, self.b = True, float(d["mm_par_px"]), float(d.get("decalage_diametre_mm", 0.0))
            if d.get("camera_matrix") is not None:
                self.K, self.dist = np.array(d["camera_matrix"], float), np.array(d["dist_coeffs"], float)

    def enregistrer(self, chemin, refs):
        """
        refs = liste de (diamètre mesuré en px, diamètre vrai en mm) des étalons.
        1 étalon  → D_mm = a·D_px           (a = mm par pixel)
        ≥ 2       → D_mm = a·D_px + b       (droite des moindres carrés : b corrige le petit biais du bord)
        Écrit calibration.json (même format que calibration_mm.py) et l'utilise tout de suite.
        """
        Dp = np.array([r[0] for r in refs]); Dm = np.array([r[1] for r in refs])
        if len(refs) >= 2 and np.ptp(Dm) > 0:
            a, b = np.polyfit(Dp, Dm, 1)                              # droite qui passe au plus près des étalons
        else:
            a, b = float(np.mean(Dm / Dp)), 0.0                       # simple règle de trois
        d = json.load(open(chemin, encoding="utf-8")) if os.path.exists(chemin) else {}
        d.update(mm_par_px=float(a), decalage_diametre_mm=float(b), date=time.strftime("%Y-%m-%d %H:%M"),
                 etalons=[{"px": float(p_), "mm": float(m_)} for p_, m_ in refs])
        json.dump(d, open(chemin, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        self.ok, self.a, self.b = True, float(a), float(b)
        return a, b

    def diam(self, px):
        return f"{self.a * px + self.b:.3f} mm" if self.ok else f"{px:.2f} px"

    def long(self, px):
        return f"{self.a * px:.3f} mm" if self.ok else f"{px:.2f} px"

    def vd(self, px):
        return self.a * px + self.b if self.ok else px

    def vl(self, px):
        return self.a * px if self.ok else px

    def axe(self, d):
        """Diamètre de l'oblong en mm : « diam » (bord à bord de la branche) → a·px + b ; « long » → a·px."""
        return self.vd(d["px"]) if d["type"] == "diam" else self.vl(d["px"])

    def corriger(self, img):
        """Corrige la distorsion de l'objectif si une calibration damier existe."""
        if self.K is None:
            return img
        h, w = img.shape[:2]
        if self._maps is None or self._maps[2] != (w, h):
            m1, m2 = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, (w, h), cv2.CV_16SC2)
            self._maps = (m1, m2, (w, h))
        return cv2.remap(img, self._maps[0], self._maps[1], cv2.INTER_LINEAR)


# =========================================================================================================
# 2. SOURCES D'IMAGES : caméra Basler (réglages manuels + exposition automatique maison) ou fichier
# =========================================================================================================
class SourceBasler:
    def __init__(self, index):
        from pypylon import pylon
        self.pylon = pylon
        infos = pylon.TlFactory.GetInstance().EnumerateDevices()
        if index >= len(infos):
            raise SystemExit(f"Caméra n° {index} absente ({len(infos)} trouvée(s)). Câble ? pylon Viewer fermé ?")
        self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(infos[index]))
        self.cam.Open()
        di = self.cam.GetDeviceInfo()
        print("Caméra :", di.GetModelName(), "S/N", di.GetSerialNumber())
        for nom, val in [("ExposureAuto", "Off"), ("GainAuto", "Off"), ("Gain", 0.0), ("ExposureTime", EXPO_US),
                         ("AcquisitionFrameRateEnable", True), ("AcquisitionFrameRate", 20.0)]:
            self._regler(nom, val)                                    # tout en manuel : mesures répétables
        self._regler("BalanceWhiteAuto", "Once")
        self.conv = pylon.ImageFormatConverter()
        self.conv.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.expo = EXPO_US

    def _regler(self, nom, val):
        try:
            getattr(self.cam, nom).Value = val
        except Exception:
            pass

    def lire(self):
        r = self.cam.RetrieveResult(100, self.pylon.TimeoutHandling_Return)
        img = None
        if r is not None and r.IsValid():
            if r.GrabSucceeded():
                img = self.conv.Convert(r).Array.copy()
            r.Release()
        return img

    def exposition(self, facteur):
        self.expo = float(np.clip(self.expo * facteur, 50, 200000))   # comme dans segmentation_branche.py
        self._regler("ExposureTime", self.expo)

    def fermer(self):
        try:
            self.cam.StopGrabbing(); self.cam.Close()
        except Exception:
            pass


class SourceImage:
    def __init__(self, chemin):
        self.img = cv2.imread(chemin, cv2.IMREAD_COLOR)
        if self.img is None:
            raise SystemExit("Image illisible : " + chemin)
        self.expo = None

    def lire(self):
        return self.img.copy()

    def exposition(self, facteur):
        pass

    def fermer(self):
        pass


# =========================================================================================================
# 3. SEGMENTATION (réglages fixes de segmentation_branche.py) — le masque sert de GUIDE, la mesure fine est en partie 6
# =========================================================================================================
def plus_grande_composante(m):
    """Garde la plus grosse tache blanche (sert aussi pour la tige, partie 6)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m, 0
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return np.where(lab == k, 255, 0).astype(np.uint8), int(st[k, cv2.CC_STAT_AREA])


def ecart_au_fond(img_bgr):
    """
    Pour chaque pixel : distance de sa COULEUR (B, G, R) à la couleur du FOND (0 = fond, grand = branche).
    Pourquoi : en niveaux de gris, les parties PÂLES de la branche (bois clair, reflets, bourgeon crème) ont
    presque le même gris que le fond blanc → Otsu les jetait. Leur COULEUR (jaune/beige) les distingue du blanc.
    La couleur du fond = médiane du cadre de 10 px autour de l'image (la branche n'en occupe qu'une petite partie).
    Propriété utile : un pixel de bord = mélange α·branche + (1−α)·fond → son écart = α × écart de la branche :
    l'écart varie LINÉAIREMENT à travers le bord, donc la règle des 50 % (bords sub-pixel) reste exacte.
    """
    f = img_bgr.astype(np.float32)
    cadre = np.concatenate([f[:10].reshape(-1, 3), f[-10:].reshape(-1, 3), f[:, :10].reshape(-1, 3), f[:, -10:].reshape(-1, 3)])
    fond = np.median(cadre, axis=0)                                   # couleur du fond (B, G, R)
    e = np.sqrt(((f - fond) ** 2).sum(axis=2)) / math.sqrt(3)         # distance couleur, ramenée à l'échelle 0…255
    d_cadre = np.sqrt(((cadre - fond) ** 2).sum(axis=1)) / math.sqrt(3)
    bruit_fond = float(np.percentile(d_cadre, 90))                    # variation normale du fond (bruit, ombre légère)
    return e, bruit_fond


def segmenter(e, bruit_fond, P=None):
    """
    Masque de la branche ENTIÈRE à partir de l'écart au fond e (voir ecart_au_fond), avec 2 seuils (hystérésis) :
      seuil HAUT = Otsu  → les parties sûres (écorce sombre)
      seuil BAS  = juste au-dessus du bruit du fond → les parties pâles, gardées SEULEMENT si elles touchent
                   une partie sûre (une poussière isolée sur le fond n'est pas gardée).
    Renvoie (masque rempli, contour) ou (None, None).
    """
    k = FLOU * 2 + 1
    ef = cv2.GaussianBlur(e, (k, k), 0)                               # gros flou : seulement pour trouver les parties SÛRES
    e3 = cv2.GaussianBlur(e, (3, 3), 0)                               # petit flou : pour placer le CONTOUR au bon endroit
    e8 = np.clip(ef, 0, 255).astype(np.uint8)
    haut, _ = cv2.threshold(e8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bas = min(haut, max(SEUIL_BAS_MIN, 2.0 * bruit_fond + 4))        # seuil bas, jamais au-dessus du seuil haut
    if P is not None:                                                 # mémorisé pour la fenêtre 3 (graphique d'Otsu)
        P["seuil_haut"], P["seuil_bas"] = float(haut), float(bas)
        P["histo"] = np.bincount(e8[::2, ::2].ravel(), minlength=256)  # histogramme (1 pixel sur 4 suffit)
    sur = (ef > haut).astype(np.uint8)
    possible = (e3 > bas).astype(np.uint8)                            # (le gros flou ferait « gonfler » le contour)
    n, lab = cv2.connectedComponents(possible, connectivity=8)        # morceaux « possibles »
    garder = np.unique(lab[(sur > 0) & (lab > 0)])                   # ceux qui touchent une partie sûre
    m = np.isin(lab, garder).astype(np.uint8)                         # toute la branche, parties pâles comprises
    # --- seuil ADAPTATIF à 50 % : chaque pixel est comparé à la moitié du niveau de la branche AUTOUR de lui
    #     (écorce sombre : niveau ≈ 200 → seuil ≈ 100 ; bois pâle : niveau ≈ 40 → seuil ≈ 20).
    #     Ainsi le contour tombe AU MILIEU du bord partout (ni gonflé par le seuil bas, ni rongé sur le pâle).
    r = int(max(15, 0.03 * max(e.shape)))                            # rayon de voisinage ≈ 60 px
    num = cv2.boxFilter(e3 * m, -1, (2 * r + 1, 2 * r + 1), normalize=False)
    den = cv2.boxFilter(m.astype(np.float32), -1, (2 * r + 1, 2 * r + 1), normalize=False)
    niveau = num / np.maximum(den, 1.0)                               # niveau moyen de la branche autour du pixel
    m = ((m > 0) & (e3 > np.maximum(bas, 0.5 * niveau))).astype(np.uint8) * 255
    t = MORPHO * 2 + 1
    noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, noyau)                    # supprime les poussières
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, noyau)                   # bouche les petits trous
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [c for c in cs if cv2.contourArea(c) >= AIRE_MIN]
    if not cs:
        return None, None
    c = max(cs, key=cv2.contourArea)                                  # la plus grosse forme = la branche
    plein = np.zeros_like(m)
    cv2.drawContours(plein, [c], -1, 255, cv2.FILLED)                 # branche remplie, sans trous
    return plein, c


# =========================================================================================================
# 5. REDRESSEMENT : la branche devient horizontale (pleine résolution, interpolation bicubique)
# =========================================================================================================
def redresser(g, masque):
    ys, xs = np.nonzero(masque)
    if len(xs) < 100:
        return None
    sel = np.random.default_rng(0).choice(len(xs), min(len(xs), 60000), replace=False)
    pts = np.column_stack([xs[sel], ys[sel]]).astype(float)
    c = pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov((pts - c).T))                  # ACP : direction principale = axe de la branche
    u = vecs[:, np.argmax(vals)]
    if u[0] < 0:
        u = -u                                                       # s croît de la gauche vers la droite
    v = np.array([-u[1], u[0]])
    P_all = np.column_stack([xs, ys]).astype(float) - c
    s, t = P_all @ u, P_all @ v
    marge = max(30, int(0.25 * (t.max() - t.min())) + 10)            # assez de fond autour pour mesurer les niveaux
    s0, s1, t0, t1 = s.min() - marge, s.max() + marge, t.min() - marge, t.max() + marge
    W, H = int(math.ceil(s1 - s0)) + 1, int(math.ceil(t1 - t0)) + 1
    M = np.array([[u[0], u[1], -(u @ c) - s0], [v[0], v[1], -(v @ c) - t0]])
    return {"g": cv2.warpAffine(g.astype(np.float32), M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE),
            "m": cv2.warpAffine(masque, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0),
            "Minv": cv2.invertAffineTransform(M), "angle": math.degrees(math.atan2(u[1], u[0]))}


def vers_image(R, pts):
    p = np.asarray(pts, float).reshape(-1, 2)
    return np.hstack([p, np.ones((len(p), 1))]) @ R["Minv"].T


# =========================================================================================================
# 6. BORDS SUB-PIXEL ET PROFIL DE DIAMÈTRE
# =========================================================================================================
def tige(mask_r):
    """Tige sans vrille ni pétiole : ouverture avec un disque de COUPE_APP_PCT % du diamètre (calcul en réduit)."""
    larg = (mask_r > 0).sum(axis=0)
    D0 = float(np.median(larg[larg > 0]))
    k = max(3.0, D0 * COUPE_APP_PCT / 100)
    f = min(1.0, 31.0 / k)
    p = cv2.resize(mask_r, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else mask_r
    _, p = cv2.threshold(p, 127, 255, cv2.THRESH_BINARY)
    kk = max(3, int(k * f) | 1)
    t = cv2.morphologyEx(p, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    if f < 1:
        t = cv2.resize(t, (mask_r.shape[1], mask_r.shape[0]), interpolation=cv2.INTER_LINEAR)
        _, t = cv2.threshold(t, 127, 255, cv2.THRESH_BINARY)
        t = cv2.bitwise_and(t, mask_r)
    return plus_grande_composante(t)[0], D0


def bords_subpixel(R, t, contraste_min):
    """
    Pour chaque colonne : niveau local = moitié entre le gris du FOND et celui de la BRANCHE (médianes),
    bord = endroit où le gris traverse ce niveau, interpolé entre 2 pixels (≈ 0,05 px). Toutes les colonnes d'un coup.
    """
    g, H = R["g"], R["g"].shape[0]
    col = t > 0
    pres = np.nonzero(col.any(axis=0))[0]
    sub = col[:, pres]
    yt, yb = sub.argmax(axis=0), H - 1 - sub[::-1].argmax(axis=0)
    D0 = float(np.median(yb - yt + 1))
    w = int(max(3, round(0.06 * D0)))                                  # demi-fenêtre de recherche autour du guide
    e1, e2 = int(max(3, 0.05 * D0)), int(max(8, 0.15 * D0))            # bande de mesure des niveaux (hors du flou)
    lire = lambda rows: g[np.clip(rows, 0, H - 1), pres[:, None]]
    bande, fen = np.arange(e1, e2)[None, :], np.arange(-w, w + 1)[None, :]
    out = {}
    for nom, y_ref, sens in (("haut", yt, -1), ("bas", yb, +1)):
        I_f = np.median(lire(y_ref[:, None] + sens * bande), axis=1)   # fond (extérieur)
        I_b = np.median(lire(y_ref[:, None] - sens * bande), axis=1)   # branche (intérieur)
        niv = 0.5 * (I_f + I_b)
        y0 = np.round(y_ref + 0.5 * sens).astype(int)
        d = lire(y0[:, None] + fen) - niv[:, None]
        chg = (np.sign(d[:, :-1]) * np.sign(d[:, 1:]) <= 0) & (d[:, :-1] != d[:, 1:])
        cout = np.where(chg, np.abs(np.arange(2 * w)[None, :] + 0.5 - w), np.inf)   # traversée la plus proche du guide
        kb = np.argmin(cout, axis=1); ii = np.arange(len(kb))
        g0, g1 = d[ii, kb], d[ii, kb + 1]
        pos = y0 - w + kb + np.where(g0 != g1, g0 / np.where(g0 != g1, g0 - g1, 1), 0.0)
        ok = np.isfinite(cout[ii, kb]) & (np.abs(I_f - I_b) >= contraste_min) & (y_ref - e2 >= 0) & (y_ref + e2 < H)
        out[nom] = np.where(ok, pos, np.nan)
    return pres.astype(float), out["haut"], out["bas"]


def lissage_gauss(v, sigma):
    """Moyenne pondérée gaussienne qui ignore les trous (NaN)."""
    ok = np.isfinite(v)
    out = v.copy()
    if sigma >= 0.5 and ok.sum() >= 3:
        r = int(3 * sigma)
        ker = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        num = np.convolve(np.where(ok, v, 0.0), ker, mode="same")
        den = np.convolve(ok.astype(float), ker, mode="same")
        out = np.where(den > 1e-6, num / np.maximum(den, 1e-9), np.nan)
    i, okk = np.arange(len(v)), np.isfinite(out)
    if okk.sum() >= 2 and not okk.all():
        out[~okk] = np.interp(i[~okk], i[okk], out[okk])
    return out


def hampel(v, fen, k=3.0, mini=1.0):
    """Rejette (NaN) les points trop loin de la médiane de leurs voisins : bord pris sur une vrille, poussière…"""
    h = fen // 2
    if len(v) < 2 * h + 1:
        return v.copy()
    W = np.lib.stride_tricks.sliding_window_view(np.concatenate([np.full(h, np.nan), v, np.full(h, np.nan)]), 2 * h + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(W, axis=1)
        mad = 1.4826 * np.nanmedian(np.abs(W - med[:, None]), axis=1)
    return np.where(np.abs(v - med) > np.maximum(k * np.nan_to_num(mad), mini), np.nan, v)


def profil(R, P):
    """Bords sub-pixel → rayons haut/bas autour d'une ligne centrale robuste → diamètre (symétrie si un côté est caché)."""
    t, D0 = tige(R["m"])
    s, haut, bas = bords_subpixel(R, t, P["contraste_min"])
    if len(s) < 30:
        return None
    ok2 = np.isfinite(haut) & np.isfinite(bas)
    if ok2.sum() < max(30, 0.25 * len(s)):                            # image bougée/floue : trop peu de bords nets
        return None
    centre = 0.5 * (haut + bas)
    sel = ok2.copy()
    cl = np.where(ok2, centre, np.nanmean(centre))
    for _ in range(3):                                                # ligne centrale = parabole robuste
        if sel.sum() < 10:
            break
        pp = np.polyfit(s[sel], centre[sel], 2); cl = np.polyval(pp, s)
        res = centre - cl
        sel = ok2 & (np.abs(res) < max(3 * 1.4826 * np.median(np.abs(res[sel])), 0.5))
    fen = max(5, int(0.15 * D0))
    rh, rb = hampel(cl - haut, fen), hampel(bas - cl, fen)           # rayons haut et bas, nettoyés
    okh, okb = np.isfinite(rh), np.isfinite(rb)
    D = np.where(okh & okb, rh + rb, np.where(okh, 2 * rh, np.where(okb, 2 * rb, np.nan)))  # 1 côté caché → symétrie
    # --- lissage choisi AUTOMATIQUEMENT selon le bruit des bords
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bruit = 1.4826 * np.nanmedian(np.abs(np.diff(D))) / math.sqrt(2)     # bruit colonne à colonne (px)
    bruit = float(bruit) if np.isfinite(bruit) else 0.5
    sig = float(np.clip(0.03 * D0 * max(1.0, bruit / 0.3), 0.02 * D0, 0.08 * D0))
    P["lissage_px"], P["bruit_bord_px"] = sig, bruit
    rh_f, rb_f = np.where(okh, rh, rb), np.where(okb, rb, rh)
    return {"s": s, "haut": haut, "bas": bas, "D": D, "D_lisse": lissage_gauss(D, sig),
            "haut_lisse": cl - lissage_gauss(rh_f, sig), "bas_lisse": cl + lissage_gauss(rb_f, sig),
            "centre": np.where(ok2, centre, cl), "D0": D0, "valides": 100 * np.isfinite(D).mean()}


# =========================================================================================================
# 7. NŒUD : pic sub-pixel, diamètre de base sans le nœud, limites interpolées, test « vrai nœud »
# =========================================================================================================
def _traverse(sv, L, i, niveau, pas):
    while 0 <= i + pas < len(L) and L[i + pas] > niveau:
        i += pas
    if not 0 <= i + pas < len(L):
        return float(sv[i])
    a, b = L[i], L[i + pas]
    return float(sv[i] + ((a - niveau) / (a - b) if a != b else 0.0) * (sv[i + pas] - sv[i]))


def noeud(B, P):
    sv, L, D0 = B["s"], B["D_lisse"], B["D0"]
    zone = (sv >= sv[0] + MARGE_BOUTS * D0) & (sv <= sv[-1] - MARGE_BOUTS * D0)
    if zone.sum() < 10:
        return None
    iz = np.nonzero(zone)[0]
    i = iz[int(np.argmax(L[iz]))]
    if 0 < i < len(L) - 1:                                            # sommet de la parabole par 3 points
        y0, y1, y2 = L[i - 1], L[i], L[i + 1]
        den = y0 - 2 * y1 + y2
        dx = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if den != 0 else 0.0
        s_pic, D_pic = sv[i] + dx * (sv[i + 1] - sv[i]), y1 - 0.25 * (y0 - y2) * dx
    else:
        s_pic, D_pic = sv[i], L[i]
    base, s_g, s_d = float(np.median(L[iz])), s_pic, s_pic
    for _ in range(3):                                                # base recalculée sans la zone du nœud
        niv = base + BORD_NOEUD_PCT / 100 * (D_pic - base)
        s_g, s_d = _traverse(sv, L, i, niv, -1), _traverse(sv, L, i, niv, +1)
        tg = zone & ~((sv > s_g - 0.3 * D0) & (sv < s_d + 0.3 * D0))
        if tg.sum() > 10:
            base = float(np.median(L[tg]))
    niv = base + BORD_NOEUD_PCT / 100 * (D_pic - base)
    # --- seuil AUTOMATIQUE « c'est un nœud » : le renflement doit dépasser 6 × la variation naturelle de la tige
    tg = zone & ~((sv > s_g - 0.3 * D0) & (sv < s_d + 0.3 * D0))
    var_tige = float(1.4826 * np.median(np.abs(L[tg] - np.median(L[tg])))) if tg.sum() > 10 else 0.0
    seuil = max(0.03 * base, 6 * var_tige)
    P["seuil_noeud_px"] = seuil
    return {"ok": (D_pic - base) >= seuil, "s_pic": float(s_pic), "D_pic": float(D_pic), "base": base,
            "renfl": 100 * (D_pic / base - 1), "niveau": niv, "s_g": s_g, "s_d": s_d, "longueur": s_d - s_g,
            "s_debut": float(sv[0]), "a": float(sv[iz[0]]), "b": float(sv[iz[-1]])}


# =========================================================================================================
# 7b. OBLONG autour du nœud + GRAND et PETIT diamètre (ÉTAPE 2)
#     Tout est calculé dans l'image REDRESSÉE (s = le long de la branche, t = en travers, t augmente vers le bas) :
#     les côtés gauche/droit de l'oblong sont donc EXACTEMENT perpendiculaires à la branche.
# =========================================================================================================
def oblong(B, N):
    sv = B["s"]
    m = (sv >= N["s_g"]) & (sv <= N["s_d"])                           # colonnes ENTRE les 2 limites = zone du nœud
    if m.sum() < 3:
        return None
    t_haut = float(np.nanmin(B["haut_lisse"][m]))                     # point le plus HAUT du nœud (plus petit t)
    t_bas = float(np.nanmax(B["bas_lisse"][m]))                       # point le plus BAS du nœud (plus grand t)
    L = N["s_d"] - N["s_g"]                                           # longueur de l'oblong = longueur du nœud
    H = t_bas - t_haut                                                # hauteur de l'oblong (du plus haut au plus bas)
    return {"s_g": N["s_g"], "s_d": N["s_d"], "t_haut": t_haut, "t_bas": t_bas, "L": L, "H": H,
            "rayon": 0.5 * min(L, H),                                 # arrondis : moitié du plus petit côté → « stade »
            "centre": (0.5 * (N["s_g"] + N["s_d"]), 0.5 * (t_haut + t_bas)),
            "D_pic": N["D_pic"]}                                      # diamètre max du nœud (étape 1), pour info


def diametres_oblong(O):
    """
    Les 2 diamètres de l'oblong (schéma d1 / d2), dans l'image redressée (s le long de la branche, t en travers) :
      axe « long de la branche » : de (s_g, t_c) à (s_d, t_c)     → longueur L (entre les 2 limites)
      axe « travers »             : de (s_c, t_haut) à (s_c, t_bas) → longueur H (du haut au bas du nœud)
    Ils se croisent au CENTRE (s_c, t_c) de l'oblong. Leurs 4 bouts sont SUR le contour de l'oblong
    (sur les côtés droits ou au sommet des arrondis). d1 = le plus long, d2 = le plus court.
    "type" sert à la conversion en mm : "long" (entre 2 limites) ou "diam" (entre 2 bords de la branche).
    """
    s_c, t_c = O["centre"]
    axe_s = {"px": O["L"], "p1": (O["s_g"], t_c), "p2": (O["s_d"], t_c), "type": "long", "sens": "le long de la branche"}
    axe_t = {"px": O["H"], "p1": (s_c, O["t_haut"]), "p2": (s_c, O["t_bas"]), "type": "diam", "sens": "en travers de la branche"}
    d1, d2 = (axe_s, axe_t) if O["L"] >= O["H"] else (axe_t, axe_s)
    return d1, d2


def contour_oblong(O, n=24):
    """Points (s, t) du contour de l'oblong : 4 quarts de cercle reliés → les 4 côtés droits viennent tout seuls."""
    s0, s1, t0, t1, r = O["s_g"], O["s_d"], O["t_haut"], O["t_bas"], O["rayon"]
    coins = [((s1 - r, t0 + r), -90, 0),                              # coin haut-droit
             ((s1 - r, t1 - r), 0, 90),                               # coin bas-droit
             ((s0 + r, t1 - r), 90, 180),                             # coin bas-gauche
             ((s0 + r, t0 + r), 180, 270)]                            # coin haut-gauche
    pts = []
    for (cs, ct), a0, a1 in coins:
        for a in np.linspace(math.radians(a0), math.radians(a1), n):
            pts.append((cs + r * math.cos(a), ct + r * math.sin(a)))  # point du quart de cercle
    return np.array(pts)


def analyser(img, calib):
    """Toute la chaîne pour une image. Renvoie un dictionnaire de résultats."""
    t0 = time.perf_counter()
    img = calib.corriger(img)                                         # distorsion (si calibration damier)
    P = {"contraste_min": CONTRASTE_MIN}                              # + lissage et seuil nœud ajoutés par profil()/noeud()
    res = {"img": img, "P": P, "ok": False, "msg": ""}
    e, bruit_fond = ecart_au_fond(img)                               # 0 = fond, grand = branche (sombre OU pâle)
    P["bruit_fond"] = bruit_fond
    masque, contour = segmenter(e, bruit_fond, P)                     # masque = branche entière
    res.update(masque=masque, contour=contour)
    if masque is None:
        res["msg"] = "Aucune branche trouvee"; return res
    g = e                                                             # bords sub-pixel mesurés sur l'écart NON flouté
    R = redresser(g, masque)
    B = profil(R, P) if R is not None else None
    if B is None:
        res["msg"] = "Branche floue / qui bouge / contraste trop faible"; return res
    N = noeud(B, P)
    res.update(R=R, B=B, N=N, ms=(time.perf_counter() - t0) * 1000)
    if N is None:
        res["msg"] = "Branche trop courte"
    elif not N["ok"]:
        res["msg"] = f"Pas de noeud net (renflement {N['renfl']:.1f} % = dans le bruit)"
    res["ok"] = N is not None and N["ok"]
    res["O"] = oblong(B, N) if res["ok"] else None                    # oblong seulement si un vrai nœud est trouvé
    if res["O"] is not None:
        res["O"]["d1"], res["O"]["d2"] = diametres_oblong(res["O"])      # les 2 diamètres de l'oblong (d1 ≥ d2)
    return res


# =========================================================================================================
# 8. AFFICHAGE
# =========================================================================================================
def ligne(img, pts, couleur, ep, ferme=False):
    p = np.round(np.asarray(pts) * 16).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [p], ferme, couleur, ep, cv2.LINE_AA, shift=4)      # précision 1/16 px


def bandeau(img, lignes):
    for i, l in enumerate(lignes):
        y = 36 + 34 * i
        cv2.putText(img, l, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, l, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)


TUILE_W, TUILE_H = 800, 500         # taille de chacune des 4 fenêtres à l'écran (px)


def titre(tuile, texte):
    """Bandeau gris foncé avec un titre en haut d'une tuile."""
    cv2.rectangle(tuile, (0, 0), (tuile.shape[1], 30), (50, 50, 50), -1)
    cv2.putText(tuile, texte, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return tuile


def vue_camera(res):
    """FENÊTRE 1 : image caméra + contour, oblong, 2 limites, diamètre central et son point central. Sans texte."""
    out = res["img"].copy()
    ep = max(2, int(round(out.shape[1] / TUILE_W * 2)))              # traits assez épais pour rester visibles une fois réduits
    if res.get("contour") is not None:
        cv2.drawContours(out, [res["contour"]], -1, (0, 255, 0), max(1, ep // 2), cv2.LINE_AA)   # contour (vert)
    R, B, N, O = res.get("R"), res.get("B"), res.get("N"), res.get("O")
    if O is None:
        return out
    sv = B["s"]
    ligne(out, vers_image(R, contour_oblong(O)), (0, 215, 255), ep, ferme=True)          # OBLONG (jaune)
    for s in (N["s_g"], N["s_d"]):                                                       # 2 LIMITES (rouge)
        th, tb = np.interp(s, sv, B["haut_lisse"]), np.interp(s, sv, B["bas_lisse"])
        ligne(out, vers_image(R, [(s, th - 0.15 * B["D0"]), (s, tb + 0.15 * B["D0"])]), (0, 0, 255), ep)
    for d, coul in ((O["d1"], (255, 0, 255)), (O["d2"], (255, 120, 0))):                 # d1 (violet) et d2 (bleu)
        ligne(out, vers_image(R, [d["p1"], d["p2"]]), coul, ep)
        for p in vers_image(R, [d["p1"], d["p2"]]):                                      # petites flèches = bouts du diamètre
            cv2.circle(out, tuple(int(v) for v in np.round(p * 16)), int(2.5 * ep * 16), coul, -1, cv2.LINE_AA, shift=4)
    c = vers_image(R, [O["centre"]])[0]                                                  # POINT CENTRAL = croisement d1 × d2
    res["point_centre"] = c
    cc = tuple(int(v) for v in np.round(c * 16))                                         # coordonnées × 16 (précision 1/16 px)
    cv2.circle(out, cc, int(5 * ep * 16), (0, 0, 0), -1, cv2.LINE_AA, shift=4)            # bord noir
    cv2.circle(out, cc, int(4 * ep * 16), (0, 0, 255), -1, cv2.LINE_AA, shift=4)          # point rouge
    return out


def vue_masque(res, shape):
    """FENÊTRE 2 : masque noir et blanc (blanc = branche)."""
    m = res.get("masque")
    if m is None:
        m = np.zeros(shape[:2], np.uint8)                             # pas de branche → tout noir
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def graphique_otsu(res, w=TUILE_W, h=TUILE_H):
    """
    FENÊTRE 3 : histogramme de l'« écart de couleur au fond » (0 = fond, grand = branche), hauteur en échelle log.
    Otsu choisit le seuil (rouge) qui sépare le mieux les 2 bosses : fond (à gauche) et branche (à droite).
    Le seuil bas (orange) récupère les parties pâles de la branche (bleu clair).
    """
    g = np.full((h, w, 3), 255, np.uint8)
    P = res.get("P", {})
    hist = P.get("histo")
    if hist is None:
        return g
    x0, x1, y0, y1 = 60, w - 20, 80, h - 60                          # zone du tracé
    hl = np.log10(hist.astype(float) + 1)                             # log : le fond a beaucoup plus de pixels que la branche
    hmax = max(hl.max(), 1e-6)
    X = lambda v: int(x0 + v * (x1 - x0) / 256)                       # valeur 0…255 → position x
    haut, bas = P.get("seuil_haut", 128), P.get("seuil_bas", 0)
    for v in range(256):                                              # une barre par valeur
        coul = (170, 170, 170) if v <= bas else ((230, 180, 120) if v <= haut else (40, 80, 140))
        y = int(y1 - hl[v] / hmax * (y1 - y0))
        cv2.rectangle(g, (X(v), y), (max(X(v), X(v + 1) - 1), y1), coul, -1)
    cv2.line(g, (x0, y1), (x1, y1), (0, 0, 0), 1)                     # axe horizontal
    for v in (0, 50, 100, 150, 200, 255):                             # graduations
        cv2.putText(g, str(v), (X(v) - 10, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(g, (X(haut), y0 - 10), (X(haut), y1), (0, 0, 255), 2)    # seuil d'Otsu (rouge)
    cv2.line(g, (X(bas), y0 - 10), (X(bas), y1), (0, 140, 255), 2)   # seuil bas (orange)
    cv2.putText(g, f"Otsu = {haut:.0f}", (X(haut) + 6, y0 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(g, f"bas = {bas:.0f}", (X(bas) + 6, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2, cv2.LINE_AA)
    pct = 100 * hist[int(bas) + 1:].sum() / max(hist.sum(), 1)
    cv2.putText(g, "gris = fond    bleu clair = parties pales    marron = branche sure", (x0, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(g, f"ecart de couleur au fond (0 = fond)   |   branche = {pct:.1f} % de l'image", (x0, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return g


def panneau_infos(res, calib, etat, w=TUILE_W, h=TUILE_H):
    """FENÊTRE 4 : informations de l'oblong et des diamètres du nœud (en mm si calibré, sinon en px)."""
    g = np.full((h, w, 3), 255, np.uint8)
    L = []                                                            # lignes : (texte, couleur, épaisseur) ou (nom, valeur, couleur, épaisseur)
    N, O, B, P = res.get("N"), res.get("O"), res.get("B"), res.get("P", {})
    u = "mm" if calib.ok else "px"
    if O is not None:
        d1, d2 = O["d1"], O["d2"]
        L += [("OBLONG (jaune)", (0, 140, 200), 2),
              ("d1 = GRAND diametre (violet)", f"{calib.axe(d1):9.3f} {u}", (150, 0, 150), 2),
              ("   direction", d1["sens"], (120, 120, 120), 1),
              ("d2 = PETIT diametre (bleu)", f"{calib.axe(d2):9.3f} {u}", (200, 90, 0), 2),
              ("   direction", d2["sens"], (120, 120, 120), 1),
              ("Rapport d1 / d2", f"{d1['px'] / max(d2['px'], 1e-9):9.3f}", (0, 0, 0), 1),
              ("Rayon des arrondis", f"{calib.vl(O['rayon']):9.3f} {u}", (0, 0, 0), 1),
              ("NOEUD", (0, 0, 0), 2),
              ("Renflement (diametre max / tige)", f"{N['renfl']:9.1f} %", (0, 0, 0), 1),
              ("BRANCHE", (0, 0, 0), 2),
              ("Diametre de la tige", f"{calib.vd(N['base']):9.3f} {u}", (0, 0, 0), 1),
              ("Centre du noeud (du bout gauche)", f"{calib.vl(N['s_pic'] - N['s_debut']):9.3f} {u}", (0, 0, 0), 1)]
        if res.get("point_centre") is not None:
            x, y = res["point_centre"]
            L.insert(7, ("Centre (rouge) = croisement d1 x d2", f"x = {x:.1f}   y = {y:.1f} px", (0, 0, 200), 1))
    else:
        L.append((res.get("msg") or "Pas de noeud", (0, 0, 200), 2))
    if B is not None:
        L.append((f"Qualite : bords valides {B['valides']:.0f} %, bruit {P.get('bruit_bord_px', 0):.2f} px, {res.get('ms', 0):.0f} ms",
                  (90, 90, 90), 1))
    L.append(("Mesures en mm (calibre)" if calib.ok else "En PIXELS : pas calibre -> touche c avec un etalon",
              (0, 130, 0) if calib.ok else (0, 0, 220), 2))
    for t in etat:                                                    # moyenne / calibration en cours, réglages
        L.append((t, (200, 80, 0), 1))
    y = 56
    for ligne_ in L:
        if len(ligne_) == 4:                                          # (nom, valeur, couleur, épaisseur) → 2 colonnes alignées
            t, val, coul, ep = ligne_
            cv2.putText(g, val, (330, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 if ep < 2 else 0.6, coul, ep, cv2.LINE_AA)
        else:                                                         # (texte, couleur, épaisseur) → une seule colonne
            t, coul, ep = ligne_
        cv2.putText(g, t, (14 if not t.startswith("diametre limite") else 34, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 if ep < 2 else 0.6, coul, ep, cv2.LINE_AA)
        y += 22 if ep < 2 else 27
    cv2.putText(g, "c calibrer  r effacer  a moyenne  s enregistrer  p reglages  +/- expo  q quitter", (14, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    return g


def mosaique(res, calib, etat):
    """Assemble les 4 fenêtres en une seule image (2 × 2), séparées par un trait sombre."""
    W, H = TUILE_W, TUILE_H
    t1 = titre(cv2.resize(vue_camera(res), (W, H), interpolation=cv2.INTER_AREA), "1 - Camera : oblong, 2 limites, d1 (violet), d2 (bleu), centre")
    t2 = titre(cv2.resize(vue_masque(res, res["img"].shape), (W, H), interpolation=cv2.INTER_AREA), "2 - Masque noir et blanc (blanc = branche)")
    t3 = titre(graphique_otsu(res, W, H), "3 - Methode d'Otsu : histogramme et seuils")
    t4 = titre(panneau_infos(res, calib, etat, W, H), "4 - Informations : oblong, d1 et d2")
    sep_v = np.full((H, 4, 3), 30, np.uint8)
    sep_h = np.full((4, 2 * W + 4, 3), 30, np.uint8)
    return np.vstack([np.hstack([t1, sep_v, t2]), sep_h, np.hstack([t3, sep_v, t4])])


def graphique(res, w=1200, h=600):
    g = np.full((h, w, 3), 255, np.uint8)
    B, N = res.get("B"), res.get("N")
    if B is None or N is None:
        return g
    sv, L = B["s"], B["D_lisse"]
    if not np.isfinite(L).any():                                     # profil vide → graphique blanc
        return g
    vmax = float(np.nanmax(L)) * 1.12
    X = lambda s: 60 + (s - sv[0]) * (w - 90) / max(1, sv[-1] - sv[0])
    Y = lambda v: h - 50 - v * (h - 100) / vmax
    if N["ok"]:
        cv2.rectangle(g, (int(X(N["s_g"])), 50), (int(X(N["s_d"])), h - 50), (215, 215, 255), -1)
    ok = np.isfinite(B["D"])
    for s, v in zip(sv[ok][::2], B["D"][ok][::2]):
        cv2.circle(g, (int(X(s)), int(Y(v))), 1, (170, 170, 170), -1)
    ligne(g, np.column_stack([[X(s) for s in sv], [Y(v) for v in L]]), (0, 0, 0), 2)
    for v, c in ((N["base"], (120, 120, 120)), (N["niveau"], (0, 140, 255))):
        cv2.line(g, (60, int(Y(v))), (w - 30, int(Y(v))), c, 1, cv2.LINE_AA)
    if N["ok"]:
        for s in (N["s_g"], N["s_d"]):
            cv2.line(g, (int(X(s)), 50), (int(X(s)), h - 50), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(g, "Profil de diametre : gris = mesure sub-pixel, noir = lisse, gris clair = base, orange = niveau des limites",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return g


def zoom(res, dessin, taille=800):
    N, R, B = res.get("N"), res.get("R"), res.get("B")
    if N is None:
        return np.full((taille, taille, 3), 255, np.uint8)
    c = vers_image(R, [(N["s_pic"], np.interp(N["s_pic"], B["s"], B["centre"]))])[0]
    d = int(max(N["longueur"], N["D_pic"], B["D0"]) * 0.75)            # le zoom contient tout l'oblong
    H, W = dessin.shape[:2]
    x0, y0 = int(np.clip(c[0] - d, 0, max(0, W - 2 * d))), int(np.clip(c[1] - d, 0, max(0, H - 2 * d)))
    crop = dessin[y0:y0 + 2 * d, x0:x0 + 2 * d]
    return cv2.resize(crop, (taille, taille), interpolation=cv2.INTER_NEAREST) if crop.size else np.full((taille, taille, 3), 255, np.uint8)


# =========================================================================================================
# 9. PROGRAMME PRINCIPAL
# =========================================================================================================
def main():
    arg = next((a for a in sys.argv[1:] if not a.startswith("--")), "0")
    dossier = os.path.dirname(os.path.abspath(__file__))
    calib = Calibration(os.path.join(dossier, "calibration.json"))
    print("Calibration :", f"OK ({calib.a:.5f} mm/px)" if calib.ok else "absente → mesures en pixels")
    src = SourceBasler(int(arg)) if arg.isdigit() else SourceImage(arg)
    fen = "Branche + noeud + oblong  -  q quitter"
    cv2.namedWindow(fen, cv2.WINDOW_NORMAL)
    montrer_P, moy, erreurs, dernier_moy = False, None, set(), []
    fichier_cal = os.path.join(dossier, "calibration.json")
    etalons, cal_px, ref_mm = [], None, None                          # calibration : étalons mesurés / mesure en cours
    print("Touches : c calibrer · r effacer calibration · a moyenne · s enregistrer · p réglages · "
          "+/- exposition · q quitter")
    try:
        while True:
            img = src.lire()
            if img is None:
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
                continue
            try:                                                  # une image ratée (branche qui bouge vite, flou…)
                                                                  # ne doit JAMAIS arrêter le programme
                res = analyser(img, calib)
                if cal_px is not None and res.get("B") is not None:      # CALIBRATION en cours (touche c)
                    D = res["B"]["D"]                                     # diamètre brut sub-pixel, en PIXELS
                    n = len(D); Dc = D[int(0.2 * n):int(0.8 * n)]         # on garde le milieu de l'étalon (pas les bouts)
                    if np.isfinite(Dc).sum() > 20:
                        cal_px.append(float(np.nanmedian(Dc)))            # 1 valeur par image
                    if len(cal_px) >= N_MOYENNE:
                        px, sd = float(np.mean(cal_px)), float(np.std(cal_px, ddof=1))
                        etalons.append((px, ref_mm))
                        a_, b_ = calib.enregistrer(fichier_cal, etalons)
                        print(f"Étalon Ø {ref_mm:.3f} mm = {px:.3f} px (± {sd:.3f} px sur {len(cal_px)} images)")
                        print(f"→ calibration.json enregistré : D_mm = {a_:.6f} × D_px + {b_:+.4f}   "
                              f"({len(etalons)} étalon(s) ; 1 mm = {1 / a_:.2f} px)")
                        if len(etalons) >= 2:
                            for p_, m_ in etalons:
                                print(f"   contrôle : étalon {m_:.3f} mm → mesuré {a_ * p_ + b_:.4f} mm (écart {a_ * p_ + b_ - m_:+.4f})")
                        cal_px = None
                if moy is not None and res["ok"] and res["O"] is not None:   # moyenne en cours (touche a)
                    N, O = res["N"], res["O"]
                    moy.append((calib.axe(O["d1"]), calib.axe(O["d2"]), calib.vl(O["rayon"]),
                                calib.vd(N["base"]), calib.vl(N["s_pic"] - N["s_debut"])))
                    if len(moy) >= N_MOYENNE:
                        a = np.array(moy); sd = a.std(axis=0, ddof=1)
                        u = "mm" if calib.ok else "px"
                        print(f"\n=== MOYENNE sur {len(a)} images ({u}) : valeur ± écart-type (répétabilité) ===")
                        dernier_moy = [f"MOYENNE {len(a)} images ({u}) : valeur +/- ecart-type"]
                        for nom, m_, s_ in zip(("d1 (grand diametre)", "d2 (petit diametre)", "rayon arrondis", "diametre tige", "position noeud"), a.mean(0), sd):
                            print(f"  {nom:20s} = {m_:10.4f} ± {s_:.4f}")
                            if nom.startswith("d"):
                                dernier_moy.append(f"  {nom} = {m_:.3f} +/- {s_:.3f}")
                        moy = None
                etat = list(dernier_moy)                                  # texte en plus pour la fenêtre 4
                if moy is not None:
                    etat = [f"Moyenne en cours : {len(moy)}/{N_MOYENNE} - ne bouge pas la branche"]
                if cal_px is not None:
                    etat.append(f"CALIBRATION etalon {ref_mm:.3f} mm : {len(cal_px)}/{N_MOYENNE} - ne bouge pas")
                if montrer_P:
                    P = res["P"]
                    etat.append(f"[reglages] flou {2 * FLOU + 1}, morpho {2 * MORPHO + 1}, contraste min {CONTRASTE_MIN}, "
                                f"bord noeud {BORD_NOEUD_PCT:.0f} %, lissage {P.get('lissage_px', 0):.1f} px")
                dessin = vue_camera(res)                                  # image pleine résolution (pour s)
                ecran = mosaique(res, calib, etat)                        # les 4 fenêtres
                cv2.imshow(fen, ecran)
            except Exception as e:
                nom = f"{type(e).__name__}: {e}"
                if nom not in erreurs:                            # chaque erreur n'est écrite qu'une fois
                    erreurs.add(nom); print("[image ignorée]", nom)
                aff = cv2.resize(img, (2 * TUILE_W + 4, 2 * TUILE_H + 4))
                bandeau(aff, ["Image ignoree (branche qui bouge / floue)"])
                cv2.imshow(fen, aff)
                res, dessin, ecran = None, img, aff
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('p'):
                montrer_P = not montrer_P
            elif k == ord('c'):                                       # CALIBRATION : étalon de diamètre connu
                try:
                    ref_mm = float(input("Place l'étalon (tige calibrée) à la place de la branche.\n"
                                         "Diamètre de l'étalon en mm (ex. 10.00) : ").replace(",", "."))
                    cal_px = []; print(f"Mesure de l'étalon sur {N_MOYENNE} images…")
                except ValueError:
                    print("Nombre invalide, calibration annulée.")
            elif k == ord('r'):                                       # repartir de zéro
                etalons, cal_px = [], None
                calib.ok, calib.a, calib.b = False, 1.0, 0.0
                if os.path.exists(fichier_cal):
                    d = json.load(open(fichier_cal, encoding="utf-8"))
                    for cle in ("mm_par_px", "decalage_diametre_mm", "etalons"):
                        d.pop(cle, None)
                    json.dump(d, open(fichier_cal, "w", encoding="utf-8"), indent=2)
                print("Calibration effacée → mesures en pixels.")
            elif k in (ord('+'), ord('=')):
                src.exposition(1.5); print(f"Exposition {src.expo / 1000:.2f} ms")
            elif k == ord('-'):
                src.exposition(1 / 1.5); print(f"Exposition {src.expo / 1000:.2f} ms")
            elif k == ord('a'):
                moy = []; print(f"Moyenne sur {N_MOYENNE} images…")
            elif k == ord('s'):
                st = time.strftime("%Y%m%d_%H%M%S"); base = os.path.join(dossier, f"oblong_{st}")
                cv2.imwrite(base + "_image.png", img); cv2.imwrite(base + "_resultat.png", dessin)
                cv2.imwrite(base + "_4fenetres.png", ecran)
                f = os.path.join(dossier, "mesures_oblong.csv"); neuf = not os.path.exists(f)
                with open(f, "a", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh, delimiter=";")
                    if neuf:
                        w.writerow(["date", "unite", "d1_grand_diametre", "d1_direction", "d2_petit_diametre", "d2_direction",
                                    "rapport_d1_d2", "rayon_arrondis", "centre_x_px", "centre_y_px", "diametre_tige",
                                    "renflement_pct", "position_noeud"])
                    N, O = (res.get("N"), res.get("O")) if res else (None, None)
                    if O is not None:
                        cx, cy = res.get("point_centre", (np.nan, np.nan))
                        w.writerow([st, "mm" if calib.ok else "px",
                                    f"{calib.axe(O['d1']):.4f}", O["d1"]["sens"], f"{calib.axe(O['d2']):.4f}", O["d2"]["sens"],
                                    f"{O['d1']['px'] / O['d2']['px']:.4f}", f"{calib.vl(O['rayon']):.4f}", f"{cx:.2f}", f"{cy:.2f}",
                                    f"{calib.vd(N['base']):.4f}", f"{N['renfl']:.2f}", f"{calib.vl(N['s_pic'] - N['s_debut']):.4f}"])
                    else:
                        print("Pas d'oblong sur cette image : rien écrit dans le CSV.")
                print("Enregistré :", base + "_*.png  +", f)
    finally:
        src.fermer()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()