# Tridom Solver (0–5, 76 Steine)

Dieses Repository untersucht eine klassische, aber überraschend tiefe Frage zum **Tridom / Triomino-Spiel**:

> **Kann man mit allen 76 Tridom-Steinen (0–5) genau einmal eine zusammenhängende, möglichst kompakte 2D-Fläche legen?**

Die Berechnung erfolgt **vollautomatisch** über **GitHub Actions** – kein lokales Ausführen notwendig.

---

## 🧩 Verwendeter Steinesatz

- Ziffern: **0–5**
- Jeder Stein ist ein **gleichseitiges Dreieck**
- **Zahlen stehen an den ECKEN**
- Zwei Steine dürfen nur dann an einer Kante liegen, wenn die **beiden Eckzahlen dieser Kante übereinstimmen**

### Identität der Steine
- **Rotationen** gelten als identisch  
  (z. B. `1–2–3 ≡ 2–3–1 ≡ 3–1–2`)
- **Spiegelungen** gelten als **verschieden**  
  (`1–2–3 ≠ 1–3–2`)

### Zusammensetzung
- 6 Steine `xxx`
- 30 Steine `xxy`
- 40 Steine `xyz` (jeweils **CW / CCW** unterschiedlich)

➡️ **Gesamt: 76 unterschiedliche Steine**

---

## 🎯 Ziel der Untersuchung

Gesucht ist eine Belegung, die:

1. **alle 76 Steine genau einmal** verwendet  
2. eine **zusammenhängende Fläche** bildet  
3. **möglichst geringen Umfang** hat (nahe an einer hexagonalen Form)  
4. an **allen Kanten korrekt matched**

---

## 🟦 Vorgehensweise

Der Solver prüft automatisch zwei Zielgeometrien:

### A) Fast-Hexagon (Priorität)
- Sehr kompakte, nahezu hexagonale Fläche
- Minimaler Umfang bei 76 Dreiecken
- Anspruchsvollste, aber „schönste“ Lösung

### B) Geradlinige Alternative
- Rechteck / Parallelogramm im Dreiecksgitter
- Etwas größerer Umfang
- Mehr Freiheitsgrade → oft leichter lösbar

Der Solver versucht **zuerst A**, und **nur falls A scheitert**, wird **B** geprüft.

---

## ⚙️ Ausführung (ohne eigenen Rechner)

Die Berechnung läuft vollständig über **GitHub Actions**.

### Solver starten
1. Öffne den Tab **Actions**
2. Wähle **Tridom Solver**
3. Klicke auf **Run workflow**
4. Warte, bis der Lauf beendet ist

### Ergebnis
Nach Abschluss findest du unter **Artifacts**:

- `solution_A.png / solution_A.pdf` **oder**
- `solution_B.png / solution_B.pdf`

Die Grafik zeigt die Fläche **von oben**, mit allen **Eckzahlen sichtbar**, so wie die Steine auf einem Tisch liegen würden.

---

## 📄 Dateien im Repository

- `tridom_solver.py`  
  → vollständiger Constraint-Solver (Backtracking + Propagation)

- `.github/workflows/solve.yml`  
  → GitHub Action zum automatischen Rechnen

- `README.md`  
  → diese Beschreibung

---

## 🧠 Hintergrund

Dieses Projekt ist **kein Spiel**, sondern eine **kombinatorische Untersuchung**:
- Es geht um Existenz oder Nicht-Existenz einer Belegung
- Manuelles Probieren ist praktisch aussichtslos
- Rechnergestützte Suche ist der einzig sinnvolle Weg

Ein **nicht gefundenes Ergebnis** ist **kein Beweis der Unmöglichkeit**, aber ein **starkes Indiz**.  
Ein **gefundenes Ergebnis** ist eine explizite, überprüfbare Lösung.

---

## 📜 Lizenz / Nutzung

Frei nutzbar für private und wissenschaftliche Zwecke.  
Keine Gewähr für Rechenzeit oder Ergebnis.

---

*Projektidee und Problemstellung: Daniel*  
*Umsetzung & Solverlogik: automatisiert*
