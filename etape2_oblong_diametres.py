# -*- coding: utf-8 -*-
"""
ÉTAPE 2 — OBLONG autour du nœud + GRAND et PETIT diamètre du nœud.

L'oblong est un rectangle aux bouts arrondis :
  • ses 2 côtés gauche / droit = les 2 limites parallèles du nœud (étape 1),
  • ses côtés haut / bas = tangentes au point le plus haut et le plus bas du nœud,
  • rayon des arrondis = moitié du plus petit côté (forme « stade »).
Grand diamètre = diamètre maximal du nœud (au pic, perpendiculaire à la branche).
Petit diamètre = diamètre minimal à l'intérieur du nœud (en pratique : à une des 2 limites).

Lancer :  python3 etape2_oblong_diametres.py [0 | 1 | photo.png] [--calib=calibration.json]
Explications détaillées : 02_Etape2_oblong_diametres.docx
"""
import numpy as np
import vitis_core as vc


def dessiner(res, calib):
    out = vc.dessiner_base(res)
    O, R, B = res.get("O"), res.get("R"), res.get("B")
    if O is None:
        return out
    vc._poly(out, vc.vers_image(R, vc.contour_oblong(O)), (0, 215, 255), 3, ferme=True)      # oblong (jaune)
    sv = B["s"]
    # grand diamètre (violet) : au pic
    th, tb = vc._interp(sv, B["haut_lisse"], O["s_grand"]), vc._interp(sv, B["bas_lisse"], O["s_grand"])
    vc._poly(out, vc.vers_image(R, [(O["s_grand"], th), (O["s_grand"], tb)]), (255, 0, 255), 3)
    # petit diamètre (cyan) : à la limite où le diamètre est le plus petit
    s_p = O["s_g"] if O["D_g"] <= O["D_d"] else O["s_d"]
    th, tb = vc._interp(sv, B["haut_lisse"], s_p), vc._interp(sv, B["bas_lisse"], s_p)
    vc._poly(out, vc.vers_image(R, [(s_p, th), (s_p, tb)]), (255, 255, 0), 3)
    return out


def textes(res, calib):
    N, O = res.get("N"), res.get("O")
    if N is None:
        return [res["msg"]]
    L = [f"Diametre de la tige (base) : {calib.diam(N['base'])}"]
    if O is not None:
        L += [f"GRAND diametre du noeud (violet) : {calib.diam(O['D_grand'])}",
              f"PETIT diametre du noeud (cyan)   : {calib.diam(O['D_petit'])}",
              f"  (a gauche {calib.diam(O['D_g'])}, a droite {calib.diam(O['D_d'])})",
              f"OBLONG : longueur {calib.long(O['L'])} x hauteur {calib.diam(O['H'])}"]
    return L


if __name__ == "__main__":
    vc.Application(2, "Etape 2 - oblong + diametres", dessiner, textes).run()
