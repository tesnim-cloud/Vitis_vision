# -*- coding: utf-8 -*-
"""
ÉTAPE 4 — Branche + nœud + point milieu + PÉRIGÉE.

Périgée = un POINT : le sommet du PETIT AXE de l'ellipse qui épouse le contour du nœud.
  • ellipse calculée par RANSAC + fitEllipseDirect sur les bords sub-pixel du nœud (entre les 2 limites),
  • 2 sommets possibles sur le petit axe → on garde celui du côté de la vrille (si elle est vue),
    sinon celui du côté du mandrin (réglage DEFAUT["mandrin_cote"] dans vitis_core.py).

Lancer :  python3 etape4_perigee.py [0 | 1 | photo.png] [--calib=calibration.json]
Explications détaillées : 04_Etape4_perigee.docx
"""
import math
import numpy as np
import cv2
import vitis_core as vc


def dessiner(res, calib):
    out = vc.dessiner_base(res)
    R, B, O, Mi, E, Pe = res.get("R"), res.get("B"), res.get("O"), res.get("Mil"), res.get("ell"), res.get("peri")
    if O is None:
        return out
    vc._poly(out, vc.vers_image(R, vc.contour_oblong(O)), (0, 215, 255), 2, ferme=True)
    for a in res.get("apps", []):                                                           # vrille / bourgeon
        xs, ys = a["pix"]
        pts = vc.vers_image(R, np.column_stack([xs[::3], ys[::3]]))
        col = (0, 255, 255) if a["type"] == "vrille" else (0, 160, 0)
        for p in pts:
            cv2.circle(out, (int(p[0]), int(p[1])), 1, col, -1)
    if E is not None:
        th = np.linspace(0, 2 * math.pi, 120)
        pts = [E["C"] + E["a"] * math.cos(t) * E["U"] + E["b"] * math.sin(t) * E["V"] for t in th]
        vc._poly(out, vc.vers_image(R, pts), (255, 255, 255), 2, ferme=True)              # ellipse (blanc)
        vc._poly(out, vc.vers_image(R, [E["C"] - E["b"] * E["V"], E["C"] + E["b"] * E["V"]]), (200, 200, 200), 1)
    if Mi is not None:
        vc._point(out, vc.vers_image(R, [Mi["M"]])[0], (0, 0, 255), 7)                     # point milieu (rouge)
    if Pe is not None:
        vc._point(out, vc.vers_image(R, [Pe["autre"]])[0], (128, 128, 128), 6)             # autre sommet (gris)
        vc._point(out, vc.vers_image(R, [Pe["P"]])[0], (0, 140, 255), 11)                  # PÉRIGÉE (orange)
    return out


def textes(res, calib):
    N, O, Mi, E, Pe = res.get("N"), res.get("O"), res.get("Mil"), res.get("ell"), res.get("peri")
    if N is None:
        return [res["msg"]]
    if O is None:
        return [f"Diametre de la tige (base) : {calib.diam(N['base'])}"]
    L = [f"GRAND diametre : {calib.diam(O['D_grand'])}   PETIT : {calib.diam(O['D_petit'])}"]
    if Mi is not None:
        xM, yM = vc.vers_image(res["R"], [Mi["M"]])[0]
        L.append(f"Point milieu : ({xM:.1f}, {yM:.1f}) px")
    if E is None:
        L.append("Ellipse du noeud impossible (pas assez de bords)")
        return L
    L.append(f"Ellipse : grand axe {calib.diam(2 * E['a'])}, petit axe {calib.diam(2 * E['b'])}, "
             f"ecart {E['rms_px']:.2f} px ({E['inliers']}/{E['total']} pts)")
    if Pe is not None:
        xP, yP = vc.vers_image(res["R"], [Pe["P"]])[0]
        L.append(f"PERIGEE (orange) : ({xP:.2f}, {yP:.2f}) px  - {Pe['raison']}")
        L.append(f"  a {calib.long(Pe['P'][0] - N['s_debut'])} du bout gauche, le long de la branche")
    vr = res.get("vrille")
    L.append("Vrille vue : " + ("oui, cote " + vr["cote"] if vr else "non"))
    return L


if __name__ == "__main__":
    vc.Application(4, "Etape 4 - perigee", dessiner, textes).run()
