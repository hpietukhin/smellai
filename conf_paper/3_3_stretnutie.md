# Stretnutie — poznámky (3.3.)

## Štruktúra článku

- **Section 3: System overview → premenovať na Method overview**
- Podsekcie:
  - A. Smell dependency model
  - B. Planning agent
    - B1. Best-First Search
    - B2. Greedy refactoring planner
- Doplniť článok na **5 strán** — greedy vs BFS je fajn obsah

## Obrázky a listing

- Sprehľadniť listing — nesmie pretekať
- **Dať obrázok z Markoviča** — kombinovaný model závislostí (ak existuje; positive/neg ak nie)

## Vzorec priority (PD/PZ)

- Odstrániť dvojku (`x2`) — nahradiť konštantou `k`, napr. `k = 0.5`
- Severity — dávame záväznosť na polovicu; lepšie `k` ako pevná záväznosť
- Popis: „k = 1, kde k je konštanta záväznosti"
- Zmeniť abstrakciu na sumu:
  - Odčítanie negatívnych závislostí tiež do vzorca (nie len suma pozitívnych)
  - **Navrhovaný všeobecný vzorec:**

    ```
    bad_smell_priority = w * severity * frequency + Σ(pos_out_edges) - Σ(neg_out_edges)
    ```

    kde `w = 0.5` (alebo iná konštanta), severity zo Sonaru, frequency = počet výskytov pachu

  - **Konkrétny vzorec (concrete):**

    ```
    conc_bad_smell_priority = w * severity + Σ(conc_pos_out_edges) - Σ(abstract_neg_out_edges)
    ```

    Konkrétne pozitívne závislosti (nájdené v zdrojáku) vs. abstraktné negatívne závislosti (z Markoviča)

- **Početnosť pachu** (`frequency`) nie je zatiaľ zohľadnená v algoritme — doplniť
- Zmeniť abstraktný vzorec na `PD` (nie `PZ`), odstrániť `x2`

## Závislosti — dôležité poznámky

- **Tranzitívna pozitívna závislosť** — zatiaľ sme nad tým neuvažovali
- Pozitívna závislosť v konkrétnom zdrojáku **nemusí nastať** (nevieme vopred)
  - Napr. na základe dát vieme zistiť, že pozitívna dep. nastáva v 80 % prípadov
  - Toto iba spomenúť — sme si toho vedomí, ale neriešime plne
- **Negatívne závislosti sú dôležitejšie ako pozitívne**, lebo pozitívne nemáme skutočne napočítané/vyhľadané/detegované v kóde
- Závažnosť positive/negative deps v skutočnom svete môže byť **variabilná/volatilná**

## Algoritmus — úpravy

- Prepisať algoritmus: má zmysel odstrániť **VŠETKY smelly, ktoré sú vyhodnotené v stave** (nielen jeden)
- Pracujeme nad **všeobecným vzorcom**, nie nad konkrétnym prípadom
  - Ako future work: mapovanie konkrétnych situácií a výpočet konkrétnych pozitívnych závislostí
- Premyslieť, či BFS **nespadne do lokálneho extrému / nezacyklí sa**

## Príklad do článku / slajdov

- Konkrétny príklad na algoritmy: pachy A, B, C, D — výsledok pre greedy aj BFS
- **Urobiť jeden slajd na 1 krok** — nech je všetko explicitné
- „Príklad: pachy v sieti/grafe" — premyslieť, či nemáme napočítané dopredu všetky alternatívy

## TODO

1. Zistiť, či máme v literatúre Best-First Search — nájsť skutočné použitia BFS pri refaktoringu
2. Odsimulovat BFS aj s novým vzorcom (s `frequency`, `k`, odčítaním neg.)
3. Premyslieť, či BFS nespadne do lokálneho extrému a nezacyklí sa
4. Premyslieť príklad „pachy v sieti/grafe" — či nemáme dopredu napočítané všetky alternatívy

## Ďalšie stretnutie

**Štvrtok o 9:00**
