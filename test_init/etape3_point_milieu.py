# -*- coding: utf-8 -*-
"""
ÉTAPE 3 — POINT MILIEU du nœud (dans l'oblong).

Construction :
  1. À chaque limite du nœud, on trace le diamètre de la branche (bord haut → bord bas) et on prend son CENTRE :
     C_g (limite gauche) et C_d (limite droite).
  2. La droite C_g – C_d passe par le centre de chaque côté de la branche : c'est l'axe local de la branche.
  3. Les côtés haut et bas de l'oblong ont aussi un centre (à mi-chemin entre les 2 limites) : la droite qui
     les relie est un diamètre perpendiculaire à l'axe, au milieu de l'oblong.
  4. Le POINT MILIEU M est le croisement de ces 2 droites.

Lancer :  python3 etape3_point_milieu.py [0 | 1 | photo.png] [--calib=calibration.json]
Explications détaillées : 03_Etape3_point_milieu.docx
"""
import numpy as np
import vitis_core as vc


def dessiner(res, calib):
    out = vc.dessiner_base(res)
    O, Mi, R, B = res.get("O"), res.get("Mil"), res.get("R"), res.get("B")
    if O is None or Mi is None:
        return out
    vc._poly(out, vc.vers_image(R, vc.contour_oblong(O)), (0, 215, 255), 2, ferme=True)      # oblong
    sv = B["s"]
    for s in (O["s_g"], O["s_d"]):                                                          # diamètres aux limites
        th, tb = vc._interp(sv, B["haut_lisse"], s), vc._interp(sv, B["bas_lisse"], s)
        vc._poly(out, vc.vers_image(R, [(s, th), (s, tb)]), (255, 255, 0), 2)
    ext = 0.6 * B["D0"]                                                                     # axe C_g → C_d prolongé
    A1, A2 = Mi["Cg"] - ext * Mi["u"], Mi["Cd"] + ext * Mi["u"]
    vc._poly(out, vc.vers_image(R, [A1, A2]), (255, 128, 0), 2)
    vc._poly(out, vc.vers_image(R, [Mi["G1"], Mi["G2"]]), (255, 0, 255), 3)               # diamètre ⟂ par le milieu
    for p, c in ((Mi["Cg"], (255, 255, 0)), (Mi["Cd"], (255, 255, 0))):
        vc._point(out, vc.vers_image(R, [p])[0], c, 6)
    vc._point(out, vc.vers_image(R, [Mi["M"]])[0], (0, 0, 255), 9)                         # POINT MILIEU
    return out


def textes(res, calib):
    N, O, Mi = res.get("N"), res.get("O"), res.get("Mil")
    if N is None:
        return [res["msg"]]
    if O is None or Mi is None:
        return [f"Diametre de la tige (base) : {calib.diam(N['base'])}"]
    xM, yM = vc.vers_image(res["R"], [Mi["M"]])[0]
    return [f"GRAND diametre : {calib.diam(O['D_grand'])}   PETIT : {calib.diam(O['D_petit'])}",
            f"POINT MILIEU (rouge) : x = {xM:.2f} px, y = {yM:.2f} px (image)",
            f"  a {calib.long(Mi['M'][0] - N['s_debut'])} du bout gauche, le long de la branche",
            f"Diametre au point milieu : {calib.diam(Mi['D_milieu'])}",
            f"Ecart pic - milieu : {calib.long(Mi['ecart_pic'])} (dissymetrie du noeud)"]


if __name__ == "__main__":
    vc.Application(3, "Etape 3 - point milieu", dessiner, textes).run()
