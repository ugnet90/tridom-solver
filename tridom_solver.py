name: Tridom Solver

on:
  workflow_dispatch:

jobs:
  solve:
    runs-on: ubuntu-latest
    timeout-minutes: 360

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install matplotlib

      - name: Run solver
        run: |
          python tridom_solver.py

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: tridom-results
          path: |
            *.png
            *.pdf
