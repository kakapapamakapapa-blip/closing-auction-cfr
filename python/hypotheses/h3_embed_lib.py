"""embed(): hidden-regime strategy table -> edge factors on the disclosure tree (see h3_embed.py)."""
import numpy as np


def embed(Th, tab, ids, Td, Sd, strip):
    """Edge factors on the disclosure tree Td from the hidden-regime table (tab, ids on Th)."""
    row = {Th.keys[n]["full"]: ids[n] for n in Th.dec}
    f = np.ones(Td.M)
    f[Td.cedges] = Td.eprob[Td.cedges]
    for i in (1, 2):
        e = Sd.pe[i]
        h = Sd.pp[i]
        r = np.array([row[strip(Td.keys[n]["full"])] for n in h])
        f[e] = tab[r, Td.eact[e]]
    return f
