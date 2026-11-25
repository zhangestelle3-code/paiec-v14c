# Executed validation

The frozen model.py SHA-256 is 700673cc9ea395fe72d839a52015440701823842cf57d51779428de93839b904. It is unchanged from the author-supplied submission ZIP.

On 20 November 2026 both audit and full modes passed in a fresh directory (Python 3.12.14, NumPy 2.3.5). Full mode regenerated 2210 seed-group rows, matching all archived budget losses and ALCs exactly. 376 real-data function parity checks passed. These are local historical diagnostics, not an official hidden-set rerun.

External full reproduction still requires the exact database described in data_input_manifest.json; it is not included. Audit mode verifies published loss aggregates without original texts or labels.
