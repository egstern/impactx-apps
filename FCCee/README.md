### Scripts for FCCee lattice

| File               | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| `fccee_z.madx`     | FCCee lattice file                                                   |
| `run_fccee_z.madx` | MAD-X script to read the lattice file and calculate lattice functions |
| `fccee_z.twiss`    | Output of the MAD-X script with lattice functions                    |
| `plot_fccee_z.py`  | Python script to plot lattice functions                              |

### Instructions

```
madx run_fccee_z.madx
python plot_fccee_z.py
```

