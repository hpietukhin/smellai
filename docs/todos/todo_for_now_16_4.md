# TODO — 16.4.

Zozbierané z: `3_3_stretnutie.md`, `todo_7_3.md`, `etxt.txt`

---

## Vzorec priority P(i)

- Premenovať `PD(i)` → `P(i)` (priority)
- Odstrániť `x2`, nahradiť konštantou `k`
- Frequency je súčin s dvoma sumami:

$$P(i) = \text{freq}(s_i) \cdot \left( w_1 \cdot \text{sev}(s_i) + \sum \text{pos\_out} - w_2 \cdot \sum \text{neg\_out} \right)$$

  - $w_1 = 0.33$ — normuje severity
  - $w_2 = 0.5$ — každý neg. pach má polovičnú váhu
- `pos_out` odstraňuje len 1 konkrétnu inštanciu → oprávnene nižšia váha ako `neg_out`
- **Frequency** nie je zatiaľ v algoritme — doplniť
- Vložiť do článku

## Fitness funkcia

- Nevolať "vzorec P(i)", ale **node score** v grafe
- Fitness = počet pachov v systéme (minimalizácia, nie cieľový stav)
- Cieľ: minimum pachov za minimum krokov (nie nutne 0 — `neg_out` vytvára nové)
- BFS sleduje fitness; ak klesá → pokračuje cestou
- Zistiť, či niekto podobnú fitness funkciu už používal → **literárna rešerš**

---

## Článok — štruktúra a úpravy

- Section 3 premenovať: "System overview" → **"Method overview"**
- Podsekcie: A. Smell dependency model / B. Planning agent (B1. Best-First Search, B2. Greedy planner)
- Doplniť na **5 strán** — greedy vs BFS je vhodný obsah
- Sprehľadniť listing — nesmie pretekať
- **Dať obrázok z Markoviča** — kombinovaný model závislostí (positive/neg)
- Nevolať "vzorec P(i)" — použiť "node score"

---

## Algoritmus — úpravy

- Prepisať: odstrániť **VŠETKY smelly vyhodnotené v stave** (nielen jeden)
- Pracovať nad **všeobecným vzorcom** (nie konkrétnym prípadom)
- Future work: mapovanie konkrétnych situácií + výpočet konkrétnych pozitívnych závislostí
- Premyslieť, či BFS **nezacyklí sa / nespadne do lokálneho extrému**
- Úplný graf: $\frac{N(N-1)}{2}$ — preveriť pre náš prípad

### Príklad do článku / slajdov

- Konkrétny príklad: pachy A, B, C, D → výsledok pre greedy aj BFS
- **1 slajd = 1 krok** (explicitné)
- Scenár: Big Switch + Long Parameter List + Long Method — overiť správanie vzorca

---

## Závislosti — poznámky

- Tranzitívna pozitívna závislosť — zatiaľ neriešené (spomenúť)
- Pozitívna závislosť v konkrétnom zdrojáku **nemusí nastať** (napr. nastáva v 80 % prípadov podľa dát)
- **Negatívne závislosti sú dôležitejšie** — pozitívne nie sú plne detegované v kóde
- Závažnosť dep. v skutočnom svete môže byť variabilná

---

## Prezentácia

- Graf je schopný prijať ďalšie dimenzie/atribúty:
  1. Váha pachu
  2. Označiť hrany / vrcholy farbou
- Použiť väčšie a ostrejšie zdrojáky s pachmi

---

## Research / nástroje

- Nájsť BFS v literatúre pri refaktoringu — skutočné použitia
- **RAG** — Fowler, RefactoringMiner → naragovať
- Marcus: **Roo-code** — vie refaktorovať? Pozrieť "Refaktorovanie s Roo-code" (YouTube, IntelliJ/VS Code pluginy)
- RefactoringMiner — Marcus niečo spomínal; preskúmať

---

## Otvorené otázky (unresolved)

1. Ako BFS zvláda lokálne extrémy / cykly?
2. Platí $\frac{N(N-1)}{2}$ pre náš stavový graf?
3. Má niekto v literatúre podobnú fitness funkciu (min. počet pachov)?
4. Konkrétne váhy $w_1, w_2$ — ako ich normovať podľa LOC/komplexity do budúcna?
