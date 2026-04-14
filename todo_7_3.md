# TODO 7.3 — Úpravy vzorca a fitness funkcie

## Vzorec P(i)

Premenovať z `PD(i)` na `P(i)` - priority.

frequency je súčin s dvoma sumami (pos a neg):

$$P(i) = \text{freq}(s_i) \cdot \left( w_1 \cdot \text{sev}(s_i) + \sum \text{pos\_out} - w_2 \cdot \sum \text{neg\_out} \right)$$

Váhy:
- $w_1 = 0.33$ — normuje závažnosť (severity)
- $w_2 = 0.5$ — každý negatívny pach má polovičnú váhu (do budúcna: normovať podľa LOC, komplexity atď.)

`pos_out` odstraňuje len 1 konkrétnu inštanciu pachu, preto je oprávnené, že má nižšiu váhu ako `neg_out`.

Vložiť do článku.

## Fitness funkcia

Nevolať to "vzorec P(i)", ale ohodnotenie uzla (node score) v grafe.

Fitness = počet pachov v systéme. Nejde o dosiahnutie konkrétneho cieľového stavu, ale o minimalizáciu. Cieľ je minimum pachov za minimum krokov — nie nutne 0, lebo cez `neg_out` vznikajú nové.

BFS prechádza sieťou a sleduje fitness. Ak sa znižuje, pokračuje touto cestou.

Zistiť, či niekto podobnú fitness funkciu už používal → literárna rešerš.

## Prechody grafom

Uvažovať o všeobecnom vzorci pre všetky prechody v stavovom grafe pachov.

Úplný graf: $\frac{N(N-1)}{2}$ — preveriť, či to platí pre náš prípad.

Príklad na overenie: Big Switch + Long Parameter List + Long Method.
- Odstránime len 1 konkrétny pach cez `pos_out`.
- Zároveň pridávame riziká cez `neg_out`.
- Overiť správanie vzorca v tomto scenári.

