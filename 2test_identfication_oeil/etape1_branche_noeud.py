# -*- coding: utf-8 -*-
"""
ÉTAPE 1 
"""
import numpy as np
import vitis_core as vc


def dessiner(res, calib):
    out = vc.dessiner_base(res)                                   # bords verts + limites rouges
    N, R, B = res.get("N"), res.get("R"), res.get("B")
    if N is not None and N["ok"]:
        sv = B["s"]
        th = vc._interp(sv, B["haut_lisse"], N["s_pic"])          # bord haut au pic
        tb = vc._interp(sv, B["bas_lisse"], N["s_pic"])           # bord bas au pic
        vc._poly(out, vc.vers_image(R, [(N["s_pic"], th), (N["s_pic"], tb)]), (255, 0, 255), 2)   # diamètre max
    return out


def textes(res, calib):
    N = res.get("N")
    if N is None:
        return [res["msg"]]
    L = [f"Diametre de la tige (base) : {calib.diam(N['base'])}",
         f"Bruit de bord : {calib.long(N['bruit_px'])}   bords valides : {res['B']['valides_pct']:.0f} %"]
    if N["ok"]:
        L += [f"NOEUD  diametre max : {calib.diam(N['D_pic'])}   (+{N['renfl']:.1f} %)",
              f"NOEUD  longueur (entre limites) : {calib.long(N['longueur'])}",
              f"Centre du noeud a {calib.long(N['s_pic'] - N['s_debut'])} du bout gauche"]
    return L


if __name__ == "__main__":
    vc.Application(1, "Etape 1 - branche + noeud", dessiner, textes).run()
