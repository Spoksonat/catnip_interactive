# CATNIP

Interactive X-ray phase-contrast imaging simulations (EI, GBI, SBI, Inline/PBI).

## Local setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Spoksonat/catnip_interactive/HEAD)

Binder config lives in `binder/` (`environment.yml` pins **Python 3.11**; several dependencies require Python ≥ 3.11).

Launch (always use `HEAD` or a recent commit hash so Binder does not reuse an old cached build):

https://mybinder.org/v2/gh/Spoksonat/catnip_interactive/HEAD