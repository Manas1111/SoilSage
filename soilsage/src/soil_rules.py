"""Rule-based soil status from soil test values (standard Indian Soil Health Card style ratings).

Ratings: pH classes, EC (dS/m), organic carbon (%), available N / P (Olsen) / K (kg/ha).
"""


def _rate(v, low, high):
    return "Low" if v < low else "Medium" if v <= high else "High"


def classify(ph=None, ec=None, oc=None, n=None, p=None, k=None) -> list[dict]:
    out = []
    if ph is not None:
        if ph < 5.5: s, note = "Strongly acidic", "liming needed"
        elif ph < 6.5: s, note = "Moderately acidic", "liming may help"
        elif ph <= 7.5: s, note = "Neutral", "ideal for most crops"
        elif ph <= 8.5: s, note = "Moderately alkaline", "watch P/Zn/Fe availability"
        else: s, note = "Strongly alkaline (sodic risk)", "gypsum reclamation likely needed"
        out.append({"param": "pH", "value": ph, "status": s, "note": note})
    if ec is not None:
        if ec < 1: s, note = "Normal", ""
        elif ec <= 2: s, note = "Critical for sensitive crops", "salt-tolerant crops preferred"
        elif ec <= 4: s, note = "Injurious to many crops", "needs leaching/drainage"
        else: s, note = "Saline", "reclamation needed"
        out.append({"param": "EC (dS/m)", "value": ec, "status": s, "note": note})
    if oc is not None:
        out.append({"param": "Organic carbon (%)", "value": oc,
                    "status": _rate(oc, 0.5, 0.75), "note": "add FYM/compost if low"})
    if n is not None:
        out.append({"param": "Available N (kg/ha)", "value": n, "status": _rate(n, 280, 560), "note": ""})
    if p is not None:
        out.append({"param": "Available P (kg/ha)", "value": p, "status": _rate(p, 10, 25), "note": ""})
    if k is not None:
        out.append({"param": "Available K (kg/ha)", "value": k, "status": _rate(k, 110, 280), "note": ""})
    return out


def to_context(rows: list[dict]) -> str:
    return "; ".join(f"{r['param']}={r['value']} ({r['status']})" for r in rows)
