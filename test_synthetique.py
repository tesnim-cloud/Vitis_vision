# -*- coding: utf-8 -*-
"""
test_synthetique.py — Banc d'essai SANS caméra : fabrique des images de branches dont on connaît
la VÉRITÉ (diamètres, limites du nœud, ellipse) et compare avec ce que mesure vitis_core.

Les images imitent une vraie caméra : dessin 4× plus fin puis réduction (bords partiels),
flou optique (σ = 1,2 px), bruit (σ = 3 niveaux de gris), branche inclinée, bourgeon et vrille.

Lancer :  python3 test_synthetique.py           → tableau d'erreurs + images test_*.png
"""
import math
import numpy as np
import cv2
import vitis_core as vc

SS = 4                                   # sur-échantillonnage du dessin


def branche(W=1920, H=1200, angle=-9.0, R0=90.0, A=0.28, w=70.0, L=1450.0, sc=0.45, clair=True,
            bourgeon=True, vrille=True, seed=0, flou=1.2, bruit=3.0):
    """Renvoie (image BGR, vérité). R0 = rayon de la tige (px), A = renflement relatif, w = largeur du nœud."""
    rng = np.random.default_rng(seed)
    c = np.array([W / 2, H / 2])
    u = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
    n = np.array([-u[1], u[0]])
    s = np.linspace(-L / 2, L / 2, 4000)
    s_c = -L / 2 + sc * L
    r = R0 * (1 + A * np.exp(-((s - s_c) / w) ** 2))
    # passage coordonnées image → coordonnées du grand dessin : centre du pixel j = SS·j + (SS−1)/2 ;
    # fillPoly colore aussi les pixels du bord → on rentre le polygone d'½ pixel du grand dessin de chaque côté
    o = (SS - 1) / 2
    r_in = r - 0.5 / SS
    haut = (c + np.outer(s, u) - np.outer(r_in, n)) * SS + o
    bas = (c + np.outer(s, u) + np.outer(r_in, n)) * SS + o
    fond, br = (222, 60) if clair else (60, 200)
    big = np.full((H * SS, W * SS), fond, np.uint8)
    cv2.fillPoly(big, [np.vstack([haut, bas[::-1]]).astype(np.int32)], br)
    pn = c + s_c * u
    if bourgeon:                                                     # bourgeon côté haut
        cb = (pn - n * (R0 * (1 + A) + 20)) * SS + o
        cv2.ellipse(big, tuple(cb.astype(int)), (int(38 * SS), int(30 * SS)), angle, 0, 360, br, -1)
    if vrille:                                                       # vrille fine côté bas
        pts = [pn + n * (R0 * (1 + A) * 0.8 + k * 2.2) + u * (25 * math.sin(k / 14) + 10) for k in range(160)]
        cv2.polylines(big, [(np.array(pts) * SS + o).astype(np.int32)], False, br, int(9 * SS))
    img = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), flou)
    img += rng.normal(0, bruit, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    d = math.sqrt(math.log(1 / 0.20)) * w                           # limites à 20 % de la hauteur du renflement
    verite = {"D_base": 2 * R0, "D_max": 2 * R0 * (1 + A), "longueur": 2 * d,
              "D_limite": 2 * R0 * (1 + 0.2 * A), "s_noeud_depuis_bout": sc * L, "milieu": pn}
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), verite


def cylindre(D, W=1920, H=1200, angle=4.0, L=1300, flou=1.2, bruit=3.0, seed=1):
    return branche(W, H, angle, D / 2, 0.0, 50, L, 0.5, True, False, False, seed, flou, bruit)[0]


if __name__ == "__main__":
    calib = vc.Calibration(None)
    P = dict(vc.DEFAUT)
    print(f"{'cas':28s} {'grandeur':18s} {'vérité':>9s} {'mesure':>9s} {'erreur px':>9s}")
    for nom, kw in [("incliné -9°, nœud +28 %", {}),
                    ("incliné +17°, nœud +15 %", dict(angle=17, A=0.15, seed=3)),
                    ("tige fine R0=60, nœud +35 %", dict(R0=60, A=0.35, w=50, seed=5)),
                    ("flou fort σ=2,5 px", dict(flou=2.5, seed=7))]:
        img, V = branche(**kw)
        r = vc.analyser(img, P, calib, etape=4)
        N, O, Mi = r["N"], r["O"], r["Mil"]
        Mi_img = vc.vers_image(r["R"], [Mi["M"]])[0]
        lignes = [("D tige", V["D_base"], N["base"]), ("D max nœud", V["D_max"], N["D_pic"]),
                  ("longueur nœud", V["longueur"], N["longueur"]), ("petit D (limite)", V["D_limite"], O["D_petit"]),
                  ("milieu x (image)", V["milieu"][0], Mi_img[0]), ("milieu y (image)", V["milieu"][1], Mi_img[1])]
        for g, vt, m in lignes:
            print(f"{nom:28s} {g:18s} {vt:9.2f} {m:9.2f} {m - vt:+9.2f}")
        cv2.imwrite(f"test_{nom.split(',')[0].replace(' ', '_').replace('°', '')}.png", img)
    # calibration avec 2 étalons puis mesure d'un 3e
    print("\nCalibration : étalons de 120 px et 240 px (vérité), mesure d'un cylindre de 180 px")
    pts = []
    for D in (120.0, 240.0):
        r = vc.analyser(cylindre(D), P, calib, etape=1)
        B = r["B"]; n = len(B["D"]); Dp = float(np.nanmedian(B["D"][int(.2 * n):int(.8 * n)]))
        pts.append((D, Dp)); print(f"  étalon {D:.1f} → mesuré {Dp:.3f} px")
    a, b = np.polyfit([p[1] for p in pts], [p[0] for p in pts], 1)
    r = vc.analyser(cylindre(180.0, seed=9), P, calib, etape=1)
    B = r["B"]; n = len(B["D"]); Dp = float(np.nanmedian(B["D"][int(.2 * n):int(.8 * n)]))
    print(f"  cylindre 180,00 → brut {Dp:.3f} px → corrigé {a * Dp + b:.3f} (a = {a:.5f}, b = {b:+.3f})")
