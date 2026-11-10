# -*- coding: utf-8 -*-
"""
calibration_mm.py — Conversion PIXELS → MILLIMÈTRES pour la caméra 1 (écrit calibration.json).

Méthode principale : un CYLINDRE ÉTALON de diamètre connu (goupille rectifiée, foret, piges, tige calibrée)
posé EXACTEMENT où sera la branche (même distance à la caméra, même éclairage).
Le programme mesure son diamètre en pixels avec le MÊME algorithme que la branche (bords sub-pixel),
puis calcule :
    • 1 étalon   : a = D_mm / D_px                      (b = 0)
    • 2 étalons + : droite D_mm = a·D_px + b (moindres carrés) → b corrige le biais de bord (éclairage, flou)

Lancer :
    python3 calibration_mm.py 0 --ref=10.00          (caméra 0, étalon de 10,00 mm)
Touches :
    m  : mesurer l'étalon (moyenne sur 20 images, ne pas bouger)  → ajoute un point
    r  : changer le diamètre de l'étalon (tapé dans le terminal)
    k  : calculer et ENREGISTRER calibration.json
    +/-: exposition      q : quitter

Option distorsion (facultatif, à faire AVANT les étalons) :
    python3 calibration_mm.py 0 --damier=9x6 --carre=5.0
    ESPACE : capturer une vue du damier (15 vues sous différents angles) · k : calculer la distorsion
Explications : 01_Etape1_branche_noeud.docx (partie « Passer des pixels aux millimètres »)
"""
import os, sys, json, time
import numpy as np
import cv2
import vitis_core as vc


def arg(nom, defaut=None):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{nom}=")), defaut)


def mesurer_cylindre(img, P, calib):
    """Diamètre du cylindre en px : médiane du profil sub-pixel sur les 60 % centraux (bouts exclus)."""
    img = calib.corriger(img)
    masque, c = vc.segmenter(img, P)
    if c is None:
        return None
    R = vc.redresser(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), masque)
    if R is None:
        return None
    B = vc.profil(R, P)
    if B is None:
        return None
    n = len(B["s"]); a, b = int(0.2 * n), int(0.8 * n)
    D = B["D"][a:b]; D = D[np.isfinite(D)]
    if len(D) < 20:
        return None
    return float(np.median(D)), float(np.std(D)), R["angle"]


def main():
    P = dict(vc.DEFAUT)
    src_arg = next((a for a in sys.argv[1:] if not a.startswith("--")), "0")
    sortie = arg("sortie", "calibration.json")
    old = json.load(open(sortie, encoding="utf-8")) if os.path.exists(sortie) else {}
    calib = vc.Calibration(sortie)
    calib.ok, calib.a, calib.b = False, 1.0, 0.0                  # on mesure en pixels BRUTS (pas d'ancienne échelle)
    src = vc.ouvrir_source(src_arg, P)
    damier = arg("damier")
    fen = "Calibration (q = quitter)"
    cv2.namedWindow(fen, cv2.WINDOW_NORMAL)
    ref = float(arg("ref", "10.0"))
    points, vues = [], []                                         # (D_mm, D_px) ; vues du damier
    if damier:
        nx, ny = map(int, damier.lower().split("x")); carre = float(arg("carre", "5.0"))
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * carre
    try:
        while True:
            img = src.lire()
            if img is None:
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
                continue
            aff = img.copy()
            if damier:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ok, coins = cv2.findChessboardCorners(g, (nx, ny), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ok:
                    cv2.drawChessboardCorners(aff, (nx, ny), coins, ok)
                lignes = [f"DAMIER {nx}x{ny}, carre {carre} mm : {len(vues)} vues (15 conseillees)",
                          "ESPACE capturer  k calculer  q quitter"]
            else:
                m = mesurer_cylindre(img, P, calib)
                lignes = [f"ETALON : {ref:.3f} mm   points enregistres : {len(points)}",
                          ("Mesure : %.3f px (+/- %.3f)  angle %.1f deg" % m) if m else "Etalon non trouve",
                          "m mesurer  r changer etalon  k enregistrer  q quitter"]
                for (dm, dp) in points:
                    lignes.append(f"  {dm:.3f} mm -> {dp:.3f} px  ({dm / dp:.5f} mm/px)")
            vc.bandeau(aff, lignes)
            cv2.imshow(fen, aff)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k in (ord('+'), ord('=')):
                src.exposition(1.25)
            elif k == ord('-'):
                src.exposition(0.8)
            elif k == ord('r'):
                ref = float(input("Diamètre de l'étalon (mm) : ").replace(",", "."))
            elif k == ord(' ') and damier and ok:
                c2 = cv2.cornerSubPix(g, coins, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
                vues.append(c2); print(f"Vue {len(vues)} capturée")
            elif k == ord('m') and not damier:
                vals = []
                print("Mesure sur 20 images, ne bouge pas l'étalon…")
                while len(vals) < 20:
                    im = src.lire()
                    if im is None:
                        cv2.waitKey(1); continue
                    r = mesurer_cylindre(im, P, calib)
                    if r:
                        vals.append(r[0])
                    cv2.waitKey(1)
                v = np.array(vals)
                points.append((ref, float(v.mean())))
                print(f"Étalon {ref:.3f} mm = {v.mean():.3f} px  (répétabilité ± {v.std(ddof=1):.3f} px)")
            elif k == ord('k'):
                d = dict(old)
                if damier:
                    if len(vues) < 8:
                        print("Il faut au moins 8 vues du damier."); continue
                    h, w = img.shape[:2]
                    rms, K, dist, _, _ = cv2.calibrateCamera([objp] * len(vues), vues, (w, h), None, None)
                    d.update(camera_matrix=K.tolist(), dist_coeffs=dist.ravel().tolist(), rms_damier_px=rms)
                    d.pop("mm_par_px", None); d.pop("decalage_diametre_mm", None)   # l'échelle doit être refaite après
                    print(f"Distorsion calculée (erreur {rms:.3f} px). REFAIS maintenant la calibration avec l'étalon.")
                else:
                    if not points:
                        print("Mesure d'abord au moins un étalon (touche m)."); continue
                    Dm = np.array([p[0] for p in points]); Dp = np.array([p[1] for p in points])
                    if len(set(np.round(Dm, 3))) >= 2:
                        a, b = np.polyfit(Dp, Dm, 1)                       # D_mm = a·D_px + b
                        res = Dm - (a * Dp + b)
                        info = f"{len(points)} étalons, droite D_mm = {a:.6f}·D_px + {b:.4f}, résidu max {np.abs(res).max():.4f} mm"
                    else:
                        a, b = float(np.mean(Dm / Dp)), 0.0
                        info = f"{len(points)} mesure(s) d'un étalon de {Dm[0]:.3f} mm"
                    d.update(mm_par_px=float(a), decalage_diametre_mm=float(b), info=info,
                             date=time.strftime("%Y-%m-%d %H:%M"), points=[list(p) for p in points])
                    print("Calibration :", info)
                with open(sortie, "w", encoding="utf-8") as f:
                    json.dump(d, f, indent=2, ensure_ascii=False)
                print("Enregistré :", os.path.abspath(sortie))
    finally:
        src.fermer()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
